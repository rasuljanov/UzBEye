import json
import math
import random
from pathlib import Path
from time import perf_counter

import torch
from torchvision.ops import generalized_box_iou, sigmoid_focal_loss

from src.common import digest, environment, peak_mib, sync, write_json
from src.data.dataset import open_image, training_input_hash
from src.data.preprocessing import corners, normalized_box
from src.evaluation.metrics import overlaps


def grounding_loss(logits, boxes, target):
    # Exact single-target assignment: the minimum-cost query is the entire Hungarian match.
    target = boxes.new_tensor(target)[None]
    distance = (boxes - target).abs().sum(-1)
    giou = generalized_box_iou(corners(boxes), corners(target))[:, 0]
    with torch.no_grad():
        matched = (5*distance - 2*giou - logits.sigmoid().mean(-1)).argmin()
    labels = torch.zeros_like(logits)
    labels[matched] = 1
    focal = sigmoid_focal_loss(logits, labels, alpha=.25, gamma=2, reduction="sum") / logits.shape[1]
    return focal + 5*distance[matched] + 2*(1-giou[matched])


def validation(model, root, rows):
    model.eval()
    correct = 0
    for row in rows:
        boxes, _ = model.predict(open_image(root, row), row["uz"])
        correct += overlaps(row, boxes)[0] >= .5
    return correct / len(rows)


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
    best, stale, updates = -1, 0, 0
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
                    loss = grounding_loss(logits, boxes, normalized_box(row))
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
            score = validation(model, root, val)
            print(json.dumps(dict(epoch=epoch+1, val_accuracy=score)), file=log, flush=True)
            print(f"{epoch_label} | Validation accuracy at IoU >= 0.5: {score:.1%}", flush=True)
            if score > best:
                best, stale = score, 0
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
               best_val_accuracy=best, epochs_completed=epoch+1, updates=updates, peak_mib=peak_mib(device),
               delta_bytes=(folder/"best.pt").stat().st_size))
    print(f"Training complete. Best validation accuracy: {best:.1%}. Checkpoint: {folder/'best.pt'}", flush=True)
