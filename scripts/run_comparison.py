import argparse
from pathlib import Path
from textwrap import fill
import torch
import pandas as pd
from PIL import ImageDraw

from src.common import config, read_json, write_json, digest
from src.data.dataset import open_image, training_input_hash
from src.evaluation.evaluate import evaluate
from src.evaluation.metrics import crossover


def draw_prediction(root, row, prediction):
    image = open_image(root, row)
    draw = ImageDraw.Draw(image)
    x, y, w, h = row["bbox"]
    draw.rectangle((x, y, x+w, y+h), outline="lime", width=4)
    a, b, c, d = prediction["boxes"][0]
    draw.rectangle((a*image.width, b*image.height, c*image.width, d*image.height), outline="red", width=2)
    return image


def visualize(out, table, records, rows, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = out/"visualizations"
    folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8,5))
    markers = {"A-cold":"o", "A-cached":"s", "B":"^", "C":"x", "D":"+"}
    for r in table:
        x, y = r["total_ms_p50"], r["accuracy_50"]
        ax.scatter(x, y, label=r["approach"], marker=markers[r["approach"]], s=65)
    front = []
    for r in sorted(table, key=lambda s:(s["total_ms_p50"], -s["accuracy_50"])):
        if not front or r["accuracy_50"] > front[-1][1]:
            front.append((r["total_ms_p50"],r["accuracy_50"]))
    if len(front) == 1:
        ax.scatter(*front[0], facecolors="none", edgecolors="gray", s=150, label="Pareto frontier")
    else:
        ax.plot(*zip(*front), color="gray", linestyle="--", zorder=0, label="Pareto frontier")
    ax.legend(loc="upper right", frameon=False)
    ax.set(xlabel="Median end-to-end latency (ms)", ylabel="Accuracy at IoU ≥ 0.5",
           ylim=(-.05,1.1), title=f"RefCOCOg UMD · {len(rows)} {rows[0]['split']} expressions")
    fig.tight_layout()
    fig.savefig(folder/"pareto.png", dpi=150)
    plt.close(fig)
    failures = []
    for tag, recs in records.items():
        for row, rec in zip(rows, recs, strict=True):
            if rec["iou"][0] < .5:
                failures.append(dict(approach=tag, id=row["id"], en=row["en"], uz=row["uz"],
                                     actual_prompt=rec["prompt"], top_iou=rec["iou"][0], tags=row.get("tags",[])))
        indices = {min(range(len(rows)), key=lambda i:recs[i]["iou"][0]),
                   max(range(len(rows)), key=lambda i:recs[i]["iou"][0])}
        for i in indices:
            row, rec = rows[i], recs[i]
            draw_prediction(root, row, rec).save(folder/f"{tag}-{row['id']}.jpg")
    write_json(out/"failures.json", failures)

    if "B" in records and "C" in records:
        differences = [c["iou"][0] - b["iou"][0] for b, c in zip(records["B"], records["C"], strict=True)]
        examples = list(dict.fromkeys([differences.index(max(differences)), differences.index(min(differences))]))
        approaches = {"A-cold": "A · Translate-Test", "B": "B · Translate-Train",
                      "C": "C · TahrirchiBERT", "D": "D · Zero-Shot Uzbek"}
        fig = plt.figure(figsize=(13, 2.9*len(examples)+.65), layout="constrained")
        fig.suptitle("Largest B/C differences in top-box IoU\nGreen = reference box  ·  Red = predicted box")
        groups = fig.subfigures(len(examples), 1, squeeze=False).ravel()
        for group, idx in zip(groups, examples, strict=True):
            row = rows[idx]
            group.suptitle(fill(f"{row['id']} · {row['uz']}", 115), fontsize=11)
            axes = group.subplots(1, len(approaches))
            for ax, (tag, label) in zip(axes, approaches.items(), strict=True):
                rec = records[tag][idx]
                ax.imshow(draw_prediction(root, row, rec))
                ax.set_title(f"{label}\nIoU {rec['iou'][0]:.2f}")
                ax.set_axis_off()
        fig.savefig(folder/"model-examples.jpg", dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Evaluate matched RefCOCOg queries sequentially, batch size one")
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--manifest", type=Path, help="Evaluation manifest. Defaults to bilingual.json under --root")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--validation", "--smoke", dest="smoke", action="store_true",
                   help="Use machine-translated validation expressions")
    p.add_argument("--hourly-usd", type=float, help="Assumed price per occupied machine-hour")
    args = p.parse_args()
    if args.warmups < 1 or (args.hourly_usd is not None and args.hourly_usd < 0):
        p.error("Require positive warm-up count and nonnegative price")
    cfg = config(args.config)
    manifest = read_json(args.manifest or args.root/"bilingual.json")
    rows = [r for r in manifest if (r["split"] == "val" if args.smoke else r["split"] == "test" and r["reviewed"])]
    if not rows:
        p.error("No eligible evaluation rows. Use a reviewed test manifest or select --validation.")
    tags = ["A-cold","A-cached","D"] + ([] if args.baselines_only else ["B","C"])
    if not args.baselines_only:
        expected = digest([r for split in ("train","val") for r in manifest if r["split"] == split])
        expected_inputs = training_input_hash(manifest)
        controls = ("seed","image_size","sequence_length","precision","epochs","accumulate","learning_rate",
                    "patience","batch_size","gradient_checkpointing","detector","detector_revision")
        b, c = [read_json(args.runs/tag/"training.json") for tag in ("B","C")]
        same_data = all(t["training_input_hash"] == expected_inputs if "training_input_hash" in t
                        else t["data_hash"] == expected for t in (b,c))
        if not same_data or any(b["config"][k]!=c["config"][k] for k in controls):
            p.error("B and C must use this training/validation manifest and matching budgets")
        if any(b["config"][k]!=cfg[k] for k in ("image_size","sequence_length","precision","detector","detector_revision")):
            p.error("Adapted and baseline inference controls differ")
    args.out.mkdir(parents=True, exist_ok=False)
    table, records = [], {}
    training_keys = {r["translation_key"] for r in manifest if r["split"] in ("train","val")}
    timing = read_json(args.root/"translation_times.json")
    preprocess = sum(timing[k] for k in training_keys)
    for tag in tags:
        run = args.runs/tag if tag in ("B","C") else None
        stats, records[tag] = evaluate(tag,args.root,rows,cfg,run,args.out,torch.device(args.device),args.warmups)
        stats["prep_seconds"] = preprocess if run else 0
        stats["machine_seconds_per_1000"] = 1000*stats["mean_seconds"]
        stats["usd_per_1000"] = (None if args.hourly_usd is None else
                                 1000*stats["mean_seconds"]*args.hourly_usd/3600)
        table.append(stats)
    costs = []
    for adapted in table:
        if adapted["approach"] in ("B","C"):
            for baseline in table[:2]:
                setup = adapted["train_seconds"] + adapted["prep_seconds"]
                costs.append(dict(adapted=adapted["approach"], baseline=baseline["approach"],
                                  break_even_queries=crossover(setup, baseline["mean_seconds"], adapted["mean_seconds"])))
    volumes = [dict(approach=r["approach"], queries=n,
                    occupied_seconds=r["train_seconds"]+r["prep_seconds"]+n*r["mean_seconds"])
               for r in table for n in (100,1000,10000,100000)]
    write_json(args.out/"costs.json", dict(hourly_usd=args.hourly_usd, comparisons=costs, volumes=volumes))
    pd.DataFrame(table).to_csv(args.out/"comparison.csv",index=False)
    write_json(args.out/"protocol.json", dict(split=rows[0]["split"], reviewed_only=not args.smoke,
               evaluation_ids=[r["id"] for r in rows], manifest_digest=digest(manifest), smoke=args.smoke))
    visualize(args.out,table,records,rows,args.root)


if __name__ == "__main__":
    main()
