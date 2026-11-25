"""Render the validation figures with the same fonts and colors as UzBEye."""

import json
from pathlib import Path

import matplotlib
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
FONT_PACKAGES = ROOT / "frontend/node_modules"
PAPER = "#2e2e26"
INK = "#faf7ee"
MUTED = "#c2beb0"
GRID = "#515044"
TEAL = "#a8cfb8"
GOLD = "#e7b38b"
SLATE = "#a9bec7"
LAVENDER = "#c9b9e5"
ROSE = "#edb8c5"
VARIANTS = {
    "A-cold": ("A · Translate-at-Inference", GOLD),
    "B-corrected": ("B · Translate-Train", TEAL),
    "C-corrected": ("C · Uzb-lang encoder", LAVENDER),
    "C-ranking": ("C · With ranking loss", ROSE),
    "D": ("D · Zero-Shot Uzbek", SLATE),
}


def load_fonts(folder):
    inter = FONT_PACKAGES / "@fontsource-variable/inter/files/inter-latin-wght-normal.woff2"
    serif = FONT_PACKAGES / "@fontsource/instrument-serif/files/instrument-serif-latin-400-normal.woff"
    if not inter.is_file() or not serif.is_file():
        raise FileNotFoundError("Install the frontend fonts with npm --prefix frontend ci.")

    with TTFont(inter) as variable:
        for weight in (400, 600):
            instance = instantiateVariableFont(variable, {"wght": weight}, updateFontNames=True)
            instance.flavor = None
            target = folder / f"inter-{weight}.ttf"
            instance.save(target)
            font_manager.fontManager.addfont(target)
    with TTFont(serif) as display_font:
        display_font.flavor = None
        title = folder / "instrument-serif.ttf"
        display_font.save(title)

    return font_manager.FontProperties(fname=title)


def finish(fig, title_font, title, subtitle, footer, out):
    fig.text(.07, .947, title, fontproperties=title_font, fontsize=29, va="top")
    fig.text(.07, .856, subtitle, fontsize=11, color=MUTED, va="top")
    fig.text(.07, .038, footer, fontsize=9, color=MUTED, va="bottom")
    fig.savefig(out, dpi=180, facecolor=PAPER)
    plt.close(fig)


def tradeoff(rows, title_font, out):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    fig.subplots_adjust(left=.14, right=.93, bottom=.20, top=.76)
    offsets = {
        "A-cold": (-10, 10), "B-corrected": (12, 14), "C-corrected": (12, -17),
        "C-ranking": (12, 10), "D": (12, 0),
    }

    for key, (label, color) in VARIANTS.items():
        row = rows[key]
        x, y = row["total_ms_p50"], 100 * row["accuracy_50"]
        ax.scatter(x, y, s=76, color=color, edgecolor=PAPER, linewidth=1.2, zorder=3)
        ax.annotate(f"{label}\n{y:.1f}% · {x:.0f} ms", (x, y), xytext=offsets[key],
                    textcoords="offset points", color=color, fontsize=11, va="center",
                    ha="right" if key == "A-cold" else "left")

    ax.set(xlim=(177, 480), ylim=(32, 53), yticks=[35, 40, 45, 50],
           xlabel="Median end-to-end latency (ms)", ylabel="Top-1 accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="both")
    finish(fig, title_font, "Accuracy versus latency",
           "C with ranking loss approaches A’s top-1 score at about half the latency.",
           "255 corrected validation prompts · Quadro RTX 5000 · A includes translation", out)


def translation_changes(rows, title_font, out):
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.3))
    fig.subplots_adjust(left=.18, right=.92, bottom=.14, top=.705, hspace=.85)
    measures = [
        ("accuracy_50", "Top-1 accuracy", (35, 46), [36, 38, 40, 42, 44, 46]),
        ("recall_5", "Top-5 recall", (74, 88), [74, 76, 78, 80, 82, 84, 86, 88]),
    ]

    for ax, (field, label, limits, ticks) in zip(axes, measures, strict=True):
        for y, model in [(1, "B"), (0, "C")]:
            before = 100 * rows[f"{model}-original"][field]
            after = 100 * rows[f"{model}-corrected"][field]
            ax.plot([before, after], [y, y], color=GRID, linewidth=3, solid_capstyle="round")
            ax.scatter(before, y, s=66, color=SLATE, edgecolor=PAPER, zorder=3)
            ax.scatter(after, y, s=72, color=TEAL, marker="D", edgecolor=PAPER, zorder=3)
            ax.annotate(f"{before:.1f}%", (before, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", color=SLATE, fontsize=11)
            ax.annotate(f"{after:.1f}%", (after, y), xytext=(0, -16), textcoords="offset points",
                        ha="center", color=TEAL, fontsize=11, fontweight=600)

        ax.set(xlim=limits, ylim=(-.62, 1.53), xticks=ticks, yticks=[0, 1],
               yticklabels=["Model C", "Model B"])
        ax.set_title(label, loc="left", fontsize=13, fontweight=600, pad=9)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="x")

    handles = [
        Line2D([], [], color=SLATE, marker="o", linestyle="none", label="Original"),
        Line2D([], [], color=TEAL, marker="D", linestyle="none", label="Corrected"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06, .811), ncol=2, frameon=False)
    finish(fig, title_font, "Effect of translation repairs",
           "Models B and C trained on the original and corrected prompts.",
           "All checkpoints evaluated on the same 255 corrected validation prompts.", out)


def ranking(rows, title_font, out):
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    fig.subplots_adjust(left=.18, right=.94, bottom=.25, top=.76)
    fills = (TEAL, GOLD, "#535147")
    labels = ("Rank 1", "Ranks 2–5", "No match in top 5")

    for y, key in enumerate(VARIANTS):
        row = rows[key]
        n, first = row["n"], row["hits"]
        top_five = round(n * row["recall_5"])
        counts = (first, top_five - first, n - top_five)
        left = 0
        for count, fill, foreground in zip(counts, fills, (PAPER, PAPER, INK), strict=True):
            ax.barh(y, count, left=left, height=.57, color=fill, edgecolor=PAPER, linewidth=.8)
            ax.text(left + count / 2, y, str(count), ha="center", va="center", color=foreground,
                    fontsize=12, fontweight=600)
            left += count

    row_labels = ["C + ranking" if key == "C-ranking" else f"Model {key[0]}" for key in VARIANTS]
    ax.set(xlim=(0, next(iter(rows.values()))["n"]), xlabel="Validation prompts",
           yticks=list(range(len(VARIANTS))), yticklabels=row_labels)
    ax.invert_yaxis()
    handles = [Patch(facecolor=fill, label=label) for fill, label in zip(fills, labels, strict=True)]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.53, .129), ncol=3, frameon=False)
    finish(fig, title_font, "Rank of the first matching box",
           "Ranking loss improves C’s first box and reduces matches in its top five.",
           "255 prompts per model · B and C use corrected training prompts", out)


def main():
    data = json.loads((ROOT / "docs/validation-results.json").read_text(encoding="utf8"))
    rows = {row["approach"]: row for row in data["results"]}
    out = ROOT / "docs/figures"
    out.mkdir(exist_ok=True)

    # FreeType can keep font files open until process exit on Windows.
    font_cache = ROOT / ".cache/chart-fonts"
    font_cache.mkdir(parents=True, exist_ok=True)
    title_font = load_fonts(font_cache)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=font_cache / "inter-400.ttf").get_name(),
        "font.size": 10.5, "figure.facecolor": PAPER, "axes.facecolor": PAPER, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": GRID, "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "axes.labelpad": 8,
    })
    tradeoff(rows, title_font, out / "accuracy-latency.png")
    translation_changes(rows, title_font, out / "translation-repair.png")
    ranking(rows, title_font, out / "box-rank-breakdown.png")



if __name__ == "__main__":
    main()
