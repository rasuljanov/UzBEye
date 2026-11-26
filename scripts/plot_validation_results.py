"""Render the validation figures with the same fonts and colors as UzBEye."""

import json
from pathlib import Path

import matplotlib
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import font_manager
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
LAVENDER = "#c9b9e5"
ROSE = "#edb8c5"
C_RUNS = ("C-corrected", "C-ranking", "C-polished-8k")
C_LINE_WIDTH = 1.35
BENCHMARK_RUNS = ("A-cold", "B-corrected", *C_RUNS)
VARIANTS = {
    "A-cold": ("A · Translate-at-Inference", GOLD, "P"),
    "B-corrected": ("B · Translate-and-Train", TEAL, "s"),
    "C-corrected": ("C-base · Uzbek-encoder-FT", LAVENDER, "D"),
    "C-ranking": ("C-4k · C-base-ranking-loss", LAVENDER, "^"),
    "C-polished-8k": ("C-8k · C-base-8k-ranking-loss", ROSE, "*"),
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
    fig.text(.06, .94, title, fontproperties=title_font, fontsize=29, va="top")
    fig.text(.06, .814, subtitle, fontsize=11, color=MUTED, va="top")
    fig.text(.06, .055, footer, fontsize=9.5, color=MUTED, va="bottom")
    fig.savefig(out, dpi=180, facecolor=PAPER)
    plt.close(fig)


def mark_run(ax, key, x, y):
    _, color, marker = VARIANTS[key]
    hollow = key == "B-corrected"
    size = 154 if marker == "*" or hollow else 110
    ax.scatter(x, y, marker=marker, s=size, facecolor=PAPER if hollow else color,
               edgecolor=color if hollow else PAPER, linewidth=1.45 if hollow else 1.05,
               zorder=3)


def chart_layout(*widths):
    fig = plt.figure(figsize=(6.8, 4.5))
    grid = fig.add_gridspec(1, len(widths) + 1, width_ratios=(1.14, *widths),
                            left=.06, right=.94, bottom=.18, top=.68, wspace=.24)
    key_ax = fig.add_subplot(grid[0])
    key_ax.set(xlim=(0, 1), ylim=(len(BENCHMARK_RUNS) - .5, -.5))
    key_ax.set_axis_off()
    key_ax.set_title("Model", loc="left", fontsize=11, fontweight=600, color=MUTED, pad=10)
    for y, key in enumerate(BENCHMARK_RUNS):
        mark_run(key_ax, key, .058, y)
        name, method = VARIANTS[key][0].split(" · ", 1)
        key_ax.annotate(name, (.145, y), xytext=(0, 5), textcoords="offset points",
                        va="center", fontsize=11, fontweight=600)
        key_ax.annotate(method, (.145, y), xytext=(0, -7), textcoords="offset points",
                        va="center", fontsize=9.5, color=MUTED)

    axes = [fig.add_subplot(grid[i], sharey=key_ax) for i in range(1, len(widths) + 1)]
    for ax in axes:
        ax.set(yticks=[])
        ax.tick_params(axis="x", pad=6, labelsize=9.5)
        ax.grid(axis="x")
    return fig, key_ax, axes


def comparison(rows, title_font, out, title, subtitle, footer, measures):
    fig, _, axes = chart_layout(1, 1)
    c_positions = [BENCHMARK_RUNS.index(key) for key in C_RUNS]
    for ax, (field, label, multiplier, ticks) in zip(axes, measures, strict=True):
        values = [multiplier * rows[key][field] for key in BENCHMARK_RUNS]
        ax.plot([values[y] for y in c_positions], c_positions, color=LAVENDER,
                linewidth=C_LINE_WIDTH, solid_capstyle="round", zorder=2)

        for y, (key, value) in enumerate(zip(BENCHMARK_RUNS, values, strict=True)):
            mark_run(ax, key, value, y)
            left = value >= .74 * ticks[-1]
            ax.annotate(f"{value:.1f}%" if multiplier == 100 else f"{value:.0f} ms",
                        (value, y), xytext=(-10 if left else 10, 0), textcoords="offset points",
                        ha="right" if left else "left", va="center", fontsize=10,
                        fontweight=600, bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1})

        ax.set(xlim=(0, ticks[-1]), xticks=ticks)
        if multiplier == 100:
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_title(label, loc="left", fontsize=11, fontweight=600, pad=10)

    finish(fig, title_font, title, subtitle, footer, out)


def tradeoff(rows, title_font, out):
    n = rows[BENCHMARK_RUNS[0]]["n"]
    comparison(rows, title_font, out, "Accuracy and latency",
               "First-box accuracy and median time per query",
               f"{n} validation prompts · Quadro RTX 5000 · A includes uncached translation",
               (("accuracy_50", "Top-1 accuracy", 100, [0, 25, 50, 75, 100]),
                ("total_ms_p50", "End-to-end latency (ms)", 1, [0, 100, 200, 300, 400, 500])))


def training_runs(rows, title_font, out):
    n = rows[BENCHMARK_RUNS[0]]["n"]
    comparison(rows, title_font, out, "Accuracy across checkpoints",
               f"Same {n} corrected Uzbek validation prompts per model",
               "The thin lines link the C variants in experiment order. IoU threshold 0.5.",
               (("accuracy_50", "Top-1 accuracy", 100, [0, 25, 50, 75, 100]),
                ("recall_5", "Top-5 recall", 100, [0, 25, 50, 75, 100])))


def ranking(rows, title_font, out):
    fig, key_ax, (ax,) = chart_layout(2.15)
    fills = (TEAL, GOLD, "#535147")
    labels = ("Rank 1", "Ranks 2–5", "Outside top 5")

    for y, key in enumerate(BENCHMARK_RUNS):
        row = rows[key]
        n, first = row["n"], row["hits"]
        top_five = round(n * row["recall_5"])
        counts = (first, top_five - first, n - top_five)
        left = 0
        for count, fill, foreground in zip(counts, fills, (PAPER, PAPER, INK), strict=True):
            ax.barh(y, count, left=left, height=.6, color=fill, edgecolor=PAPER, linewidth=.8)
            if count:
                ax.text(left + count / 2, y, str(count), ha="center", va="center", color=foreground,
                        fontsize=11, fontweight=600)
            left += count

    n = rows[BENCHMARK_RUNS[0]]["n"]
    ax.set(xlim=(0, n), xticks=[0, 50, 100, 150, 200, n], xlabel="Validation prompts")
    first, last = (BENCHMARK_RUNS.index(key) for key in (C_RUNS[0], C_RUNS[-1]))
    key_ax.plot([-.026, -.026], [first - .32, last + .32], color=LAVENDER,
                linewidth=C_LINE_WIDTH, solid_capstyle="round", clip_on=False)
    handles = [Patch(facecolor=fill, label=label) for fill, label in zip(fills, labels, strict=True)]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0,
              ncol=3, frameon=False, fontsize=10, columnspacing=1.35, handlelength=1,
              handletextpad=.5)
    finish(fig, title_font, "Rank of the first matching box",
           "Each prompt is counted once, at its earliest matching rank.",
           f"{n} prompts per run · IoU threshold 0.5", out)


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
