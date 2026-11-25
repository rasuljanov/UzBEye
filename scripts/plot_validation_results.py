"""Render the README figures from the checked-in validation results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
PAPER = "#fbf9f3"
INK = "#242d2b"
MUTED = "#4f5c57"
GRID = "#deded4"
TEAL = "#1b6a60"
RUST = "#9a4627"
SLATE = "#52656f"
LAVENDER = "#695684"
VARIANTS = {
    "A-cold": ("A · Translate at inference", RUST),
    "B-corrected": ("B · Translate-Train", TEAL),
    "C-corrected": ("C · TahrirchiBERT", LAVENDER),
    "D": ("D · Zero-Shot Uzbek", SLATE),
}


def finish(fig, title, subtitle, footer, out):
    fig.text(.10, .945, title, fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(.10, .855, subtitle, fontsize=11, color=MUTED, va="top")
    fig.text(.10, .049, footer, fontsize=9, color=MUTED, va="bottom")
    fig.savefig(out, dpi=170, facecolor=PAPER)
    plt.close(fig)


def tradeoff(rows, out):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    fig.subplots_adjust(left=.10, right=.91, bottom=.22, top=.77)
    offsets = {"A-cold": (-12, 7), "B-corrected": (13, 9), "C-corrected": (13, -12), "D": (13, 0)}
    for key, (label, color) in VARIANTS.items():
        row = rows[key]
        x, y = row["total_ms_p50"], 100 * row["accuracy_50"]
        ax.scatter(x, y, s=112, color=color, edgecolor=PAPER, linewidth=1.5, zorder=3)
        ax.annotate(f"{label}\n{y:.1f}% · {x:.0f} ms", (x, y), xytext=offsets[key],
                    textcoords="offset points", color=color, fontsize=10, va="center",
                    ha="right" if key == "A-cold" else "left")
    ax.set(xlim=(175, 482), ylim=(32, 53), yticks=[35, 40, 45, 50],
           xlabel="Median end-to-end latency (ms)", ylabel="Top-1 accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="both")
    finish(fig, "Accuracy versus latency", "A leads on top-1 accuracy. Corrected B and C take about 205 ms.",
           "255 corrected validation prompts · Quadro RTX 5000 · A includes uncached translation", out)


def translation_changes(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    fig.subplots_adjust(left=.12, right=.94, bottom=.23, top=.69, wspace=.38)
    for ax, (field, label, limits) in zip(axes, [
        ("accuracy_50", "Top-1 accuracy", (35, 46)),
        ("recall_5", "Top-5 recall", (74, 88)),
    ], strict=True):
        for y, model in [(1, "B"), (0, "C")]:
            before = 100 * rows[f"{model}-original"][field]
            after = 100 * rows[f"{model}-corrected"][field]
            ax.plot([before, after], [y, y], color=GRID, linewidth=5, solid_capstyle="round")
            ax.scatter(before, y, s=62, color=SLATE, zorder=3)
            ax.scatter(after, y, s=70, color=TEAL, marker="D", zorder=3)
            ax.annotate(f"{before:.1f}%", (before, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=SLATE, fontsize=10)
            ax.annotate(f"{after:.1f}%", (after, y), xytext=(0, -16), textcoords="offset points",
                        ha="center", color=TEAL, fontsize=10, fontweight="bold")
        ax.set(xlim=limits, ylim=(-.43, 1.42), yticks=[0, 1], yticklabels=["C", "B"])
        ax.set_title(label, loc="left", fontweight="bold", pad=9)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="x")
    fig.text(.15, .147, "●  Original translations", color=SLATE, fontsize=10)
    fig.text(.40, .147, "◆  Corrected translations", color=TEAL, fontsize=10)
    finish(fig, "What changed after translation repair?", "B gained first-ranked matches. C gained matches among its top five.",
           "All checkpoints evaluated on the same 255 corrected prompts. Dots show separate training runs.", out)


def ranking(rows, out):
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    fig.subplots_adjust(left=.23, right=.94, bottom=.25, top=.77)
    fills = (TEAL, "#e3bf80", "#e5e4dc")
    for y, key in enumerate(VARIANTS):
        row = rows[key]
        n, first = row["n"], row["hits"]
        top_five = round(n * row["recall_5"])
        counts = (first, top_five - first, n - top_five)
        left = 0
        for count, fill, foreground in zip(counts, fills, ("white", INK, INK), strict=True):
            ax.barh(y, count, left=left, height=.58, color=fill)
            ax.text(left + count / 2, y, str(count), ha="center", va="center", color=foreground,
                    fontsize=11, fontweight="bold")
            left += count
    ax.set(xlim=(0, next(iter(rows.values()))["n"]), xlabel="Validation prompts", yticks=list(range(len(VARIANTS))),
           yticklabels=[label for label, _ in VARIANTS.values()])
    ax.invert_yaxis()
    labels = ("Rank 1", "Ranks 2–5", "No match in top 5")
    handles = [Patch(facecolor=fill, label=label) for fill, label in zip(fills, labels, strict=True)]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.58, .13), ncol=3, frameon=False)
    finish(fig, "Where did the first matching box rank?", "C had 105 prompts with a match at ranks 2–5 and a miss at rank 1.",
           "Each bar contains 255 prompts. A match requires IoU ≥ 0.5. B and C use corrected training prompts.", out)


def main():
    data = json.loads((ROOT / "docs/validation-results.json").read_text(encoding="utf8"))
    rows = {row["approach"]: row for row in data["results"]}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": GRID, "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
    })
    out = ROOT / "docs/figures"
    out.mkdir(exist_ok=True)
    tradeoff(rows, out / "accuracy-latency.png")
    translation_changes(rows, out / "translation-repair.png")
    ranking(rows, out / "box-rank-breakdown.png")


if __name__ == "__main__":
    main()
