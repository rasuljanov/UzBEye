import json
import math
import random
from pathlib import Path
from time import perf_counter

import torch
from torchvision.ops import box_iou, generalized_box_iou, sigmoid_focal_loss

from src.common import digest, environment, peak_mib, sync, write_json
from src.data.dataset import open_image, training_input_hash
from src.data.preprocessing import corners, normalized_box
from src.evaluation.metrics import overlaps


def grounding_loss(logits, boxes, target, *, ranking_loss=False):
    # Exact single-target assignment: the minimum-cost query is the entire Hungarian match.
    target = boxes.new_tensor(target)[None]
    distance = (boxes - target).abs().sum(-1)
    box_corners, target_corners = corners(boxes), corners(target)
    giou = generalized_box_iou(box_corners, target_corners)[:, 0]
    scores = logits.sigmoid().mean(-1)
    with torch.no_grad():
        matched = (5*distance - 2*giou - scores).argmin()
    labels = torch.zeros_like(logits)
    labels[matched] = 1
    focal = sigmoid_focal_loss(logits, labels, alpha=.25, gamma=2, reduction="sum") / logits.shape[1]
    loss = focal + 5*distance[matched] + 2*(1-giou[matched])
    if ranking_loss:
        with torch.no_grad():
            iou = box_iou(box_corners.clamp(0, 1), target_corners)[:, 0]
            valid = iou >= .5
        # Compare the best-localized box with the highest-scoring incorrect box.
        wrong_score = scores.masked_fill(valid, -torch.inf).max()
        penalty = (.1 + wrong_score - scores[iou.argmax()]).relu()
        loss = loss + torch.where(valid.any(), penalty, 0.)
    return loss


def validation(model, root, rows):
    model.eval()
    top1 = top5 = 0
    for row in rows:
        boxes, _ = model.predict(open_image(root, row), row["uz"])
        iou = overlaps(row, boxes)
        top1 += iou[0] >= .5
        top5 += max(iou[:5]) >= .5
    return top1 / len(rows), top5 / len(rows)


def fit(model, root, rows, folder, cfg, approach):
    device = model.device
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if not train or not val or cfg["batch_size"] != 1:
        raise ValueError("Need train and val expressions and physical batch size one")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    names = {k: v for k, v in model.named_parameters() if v.requires_grad}
    opt = torch.optim.AdamW(names.values(), lr=cfg["learning_rate"], weight_decay=.01)
    accum = cfg["accumulate"]
    steps_per_epoch = math.ceil(len(train)/accum)
    steps = steps_per_epoch*cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    amp = device.type == "cuda" and cfg["precision"] == "fp16"
    scaler = torch.amp.GradScaler(device.type, enabled=amp, init_scale=128)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = perf_counter()
    best, best_top5, stale, updates = -1, 0., 0, 0
    ranking_loss = cfg.get("ranking_loss", False)
    provenance = dict(config=cfg, approach=approach, data_hash=digest(train + val),
                      training_input_hash=training_input_hash(rows),
                      environment=environment(device), trainable=sum(p.numel() for p in names.values()),
                      parameters=sum(p.numel() for p in model.parameters()))
    print(f"Training on {len(train)} expressions with {len(val)} for validation.", flush=True)
    with (folder/"training.jsonl").open("w", encoding="utf8") as log:
        for epoch in range(cfg["epochs"]):
            epoch_label = f"Epoch {epoch+1}/{cfg['epochs']}"
            print(f"{epoch_label} starting...", flush=True)
            order = list(train)
            random.Random(cfg["seed"] + epoch).shuffle(order)
            model.train()
            model.detector.model.backbone.eval()
            for start_idx in range(0, len(order), accum):
                group = order[start_idx:start_idx+accum]
                opt.zero_grad(set_to_none=True)
                total = 0.
                for row in group:
                    logits, boxes = model(open_image(root, row), row["uz"])
                    loss = grounding_loss(logits, boxes, normalized_box(row), ranking_loss=ranking_loss)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite grounding loss")
                    scaler.scale(loss / len(group)).backward()
                    total += loss.item()
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(names.values(), 1., error_if_nonfinite=not amp)
                scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                applied = scaler.get_scale() >= scale
                if applied:
                    updates += 1
                    scheduler.step()
                entry = dict(epoch=epoch+1, updates=updates, loss=total/len(group),
                             grad_norm=float(norm) if torch.isfinite(norm) else None,
                             applied=applied, lr=opt.param_groups[0]["lr"], peak_mib=peak_mib(device))
                print(json.dumps(entry), file=log, flush=True)
                step = start_idx//accum + 1
                if step == 1 or step % 10 == 0 or step == steps_per_epoch:
                    print(f"{epoch_label} | Step {step}/{steps_per_epoch} | "
                          f"Loss {entry['loss']:.4f} | Optimizer updates {updates}", flush=True)
            if updates == 0:
                raise FloatingPointError("No optimizer step applied; inspect gradient and precision logs")
            print(f"{epoch_label} | Running validation...", flush=True)
            score, top5 = validation(model, root, val)
            print(json.dumps(dict(epoch=epoch+1, val_accuracy=score, val_recall_5=top5)), file=log, flush=True)
            print(f"{epoch_label} | Validation at IoU >= 0.5 | Top-1 {score:.1%} | Top-5 {top5:.1%}", flush=True)
            if score > best:
                best, best_top5, stale = score, top5, 0
                tmp = folder/"best.tmp"
                torch.save(dict(approach=approach, config=cfg,
                                weights={k:v.detach().cpu() for k,v in names.items()}), tmp)
                tmp.replace(folder/"best.pt")
                print(f"Saved best checkpoint to {folder/'best.pt'}", flush=True)
            else:
                stale += 1
            if stale >= cfg["patience"]:
                print("Early stopping after no improvement on validation.", flush=True)
                break
    sync(device)
    write_json(folder/"training.json", provenance | dict(train_seconds=perf_counter()-start,
               best_val_accuracy=best, best_val_recall_5=best_top5, epochs_completed=epoch+1,
               updates=updates, peak_mib=peak_mib(device), delta_bytes=(folder/"best.pt").stat().st_size))
    print(f"Training complete. Best checkpoint | Top-1 {best:.1%} | Top-5 {best_top5:.1%} | {folder/'best.pt'}",
          flush=True)
