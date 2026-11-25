from math import ceil
from statistics import mean


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - w*h
    return w*h / union if union > 0 else 0.0


def overlaps(row, boxes):
    x, y, w, h = row["bbox"]
    target = [x/row["width"], y/row["height"], (x+w)/row["width"], (y+h)/row["height"]]
    return [iou(box, target) for box in boxes]


def summarize(records):
    return dict(accuracy_50=mean(r["iou"][0] >= .5 for r in records),
                recall_1=mean(r["iou"][0] >= .5 for r in records),
                recall_5=mean(max(r["iou"][:5]) >= .5 for r in records),
                mean_iou=mean(r["iou"][0] for r in records))


def crossover(setup_s, baseline_s, adapted_s):
    savings = baseline_s - adapted_s
    return ceil(setup_s/savings) if savings > 0 else None
