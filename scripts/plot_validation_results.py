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
C_RUNS = ("C-original", "C-corrected", "C-ranking", "C-polished-8k")
C_LINE_WIDTH = 1.35
BENCHMARK_RUNS = ("A-cold", "B-corrected", *C_RUNS, "D")
LEGEND_RUNS = ("A-cold", "B-corrected", "D", *C_RUNS)
VARIANTS = {
    "A-cold": ("A · Translate-at-Inference", GOLD, "P"),
    "B-original": ("B · Original", TEAL, "h"),
    "B-corrected": ("B · Translate-Train", TEAL, "s"),
    "C-original": ("C · Original", LAVENDER, "o"),
    "C-corrected": ("C · Corrected", LAVENDER, "D"),
    "C-ranking": ("C · 4k + ranking", LAVENDER, "^"),
    "C-polished-8k": ("C · Polished 8k + ranking", ROSE, "*"),
    "D": ("D · Zero-Shot Uzbek", SLATE, "X"),
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


def model_legend(fig, keys, labels=None):
    handles = []
    for key in keys:
        name, color, marker = VARIANTS[key]
        hollow = key == "B-corrected"
        handles.append(Line2D([], [], linestyle="none", marker=marker,
                              markersize=11 if marker == "*" else 9.5,
                              markerfacecolor="none" if hollow else color, markeredgewidth=1.15,
                              markeredgecolor=color, label=labels[key] if labels else name))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.053, .817), ncol=3,
               frameon=False, fontsize=9, columnspacing=1.05, handletextpad=.48)


def mark_run(ax, key, x, y, **kwargs):
    _, color, marker = VARIANTS[key]
    hollow = key == "B-corrected"
    size = 154 if marker == "*" or hollow else 110
    ax.scatter(x, y, marker=marker, s=size, facecolor="none" if hollow else color,
               edgecolor=color if hollow else PAPER, linewidth=1.45 if hollow else 1.05,
               zorder=4 if hollow else 3, **kwargs)


def tradeoff(rows, title_font, out):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    fig.subplots_adjust(left=.14, right=.93, bottom=.20, top=.598)
    c_rows = [rows[key] for key in C_RUNS]
    ax.plot([row["total_ms_p50"] for row in c_rows],
            [100 * row["accuracy_50"] for row in c_rows],
            color=LAVENDER, linewidth=C_LINE_WIDTH, solid_capstyle="round", zorder=2)

    labels = {}
    for key in BENCHMARK_RUNS:
        row = rows[key]
        x, y = row["total_ms_p50"], 100 * row["accuracy_50"]
        mark_run(ax, key, x, y)
        labels[key] = f"{VARIANTS[key][0]}\n{y:.1f}% · {x:.0f} ms"

    ax.set(xlim=(177, 480), ylim=(32, 61), yticks=[35, 40, 45, 50, 55, 60],
           xlabel="Median end-to-end latency (ms)", ylabel="Top-1 accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="both")
    # The three low-latency points are too close to separate at the full x-axis scale.
    close = ax.inset_axes([.44, .20, .38, .54])
    close.set(xlim=(202.7, 208.3), ylim=(41.0, 43.12), xticks=[204, 207], yticks=[41.5, 42.5])
    close.xaxis.set_major_formatter("{x:.0f} ms")
    close.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    close.tick_params(labelsize=8)
    close.grid(axis="both")
    for spine in close.spines.values():
        spine.set_visible(True)
    close.set_title("Close results", loc="left", fontsize=9, pad=5)
    close.plot([row["total_ms_p50"] for row in c_rows[:2]],
               [100 * row["accuracy_50"] for row in c_rows[:2]],
               color=LAVENDER, linewidth=C_LINE_WIDTH)
    for key in (*C_RUNS[:2], "B-corrected"):
        row = rows[key]
        mark_run(close, key, row["total_ms_p50"], 100 * row["accuracy_50"])

    model_legend(fig, LEGEND_RUNS, labels)
    finish(fig, title_font, "Accuracy versus latency",
           "The line follows Model C in training order.",
           "255 corrected validation prompts · Quadro RTX 5000 · A includes translation", out)


def training_runs(rows, title_font, out):
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.3), sharex=True)
    fig.subplots_adjust(left=.14, right=.94, bottom=.17, top=.683, hspace=.58)
    measures = [
        ("accuracy_50", "Top-1 accuracy", (32, 63), [35, 45, 55]),
        ("recall_5", "Top-5 recall", (49, 92), [55, 65, 75, 85]),
    ]
    b_runs = ("B-original", "B-corrected")
    b_offsets = {"B-original": .11, "B-corrected": .15}
    for ax, (field, label, limits, ticks) in zip(axes, measures, strict=True):
        b_values = [100 * rows[key][field] for key in b_runs]
        c_values = [100 * rows[key][field] for key in C_RUNS]
        ax.plot(range(len(c_values)), c_values, color=LAVENDER, linewidth=C_LINE_WIDTH,
                solid_capstyle="round")

        for keys, values, other in [(b_runs, b_values, c_values), (C_RUNS, c_values, b_values)]:
            for x, (key, value) in enumerate(zip(keys, values, strict=True)):
                position = x - b_offsets.get(key, 0)
                mark_run(ax, key, position, value)
                offset = 11 if x >= len(other) or value >= other[x] else -19
                ax.annotate(f"{value:.1f}%", (position, value), xytext=(0, offset),
                            textcoords="offset points", ha="center", color=VARIANTS[key][1], fontsize=10)

        ax.set(ylim=limits, yticks=ticks)
        ax.set_title(label, loc="left", fontsize=13, fontweight=600, pad=7)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y")

    axes[-1].set(xlim=(-.25, 3.25), xticks=[0, 1, 2, 3],
                 xticklabels=["Original", "Corrected", "4k + ranking", "Polished 8k\n+ ranking"])
    labels = {key: VARIANTS[key][0] for key in b_runs + C_RUNS}
    labels["B-corrected"] = "B · Corrected"
    model_legend(fig, b_runs + C_RUNS, labels)
    finish(fig, title_font, "Model C across training runs",
           "Each checkpoint was evaluated on the same 255 Uzbek prompts.",
           "The last two Model C runs use ranking loss.", out)


def ranking(rows, title_font, out):
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    fig.subplots_adjust(left=.15, right=.94, bottom=.235, top=.65)
    fills = (TEAL, GOLD, "#535147")
    labels = ("Rank 1", "Ranks 2–5", "No match in top 5")

    for y, key in enumerate(BENCHMARK_RUNS):
        mark_run(ax, key, -.061, y, transform=ax.get_yaxis_transform(), clip_on=False)
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

    ax.set(xlim=(0, next(iter(rows.values()))["n"]), xlabel="Validation prompts", yticks=[])
    ax.invert_yaxis()
    first, last = (BENCHMARK_RUNS.index(key) for key in (C_RUNS[0], C_RUNS[-1]))
    ax.plot([-.025, -.025], [first - .24, last + .24], transform=ax.get_yaxis_transform(),
            color=LAVENDER, linewidth=C_LINE_WIDTH, solid_capstyle="round", clip_on=False)
    model_legend(fig, LEGEND_RUNS)
    handles = [Patch(facecolor=fill, label=label) for fill, label in zip(fills, labels, strict=True)]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.53, .129), ncol=3, frameon=False)
    finish(fig, title_font, "Rank of the first matching box",
           "The polished 8k run returns the most rank-one matches.",
           "255 prompts per run · IoU threshold 0.5", out)


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
    training_runs(rows, title_font, out / "training-runs.png")
    ranking(rows, title_font, out / "box-rank-breakdown.png")



if __name__ == "__main__":
    main()
