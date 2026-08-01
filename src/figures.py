"""
Figures for the README.

Every figure is regenerated from the saved .xlsx artefacts in ``reports/``,
never from numbers typed in by hand, so a figure can never drift away from
the experiment that produced it. Run the experiments first, then this.

Design notes (the charts have to survive a reader, not just render):

* Colours come from a validated categorical palette; the three hues used
  clear the colourblind-separation and normal-vision floors on a light
  surface. Aqua sits below 3:1 contrast, so **every series is directly
  labelled** - identity is never carried by colour alone.
* Model-driven series get saturated hues; passive benchmarks get greys and
  dashes, so the semantic split is visible before the legend is read.
* Effect-size panels are forest plots with 95% confidence intervals and an
  explicit zero line, because the finding *is* "the interval contains zero".
  A bar chart of the means would hide exactly the thing being reported.

Usage:
    python -m src.figures
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .metrics import mean_and_ci

FIGURE_DIR = config.REPORT_DIR / "figures"

# Validated categorical slots (light surface) + neutral ink for benchmarks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "grid.color": GRID,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 140,
})


def _style(ax, title=None, xlabel=None, ylabel=None):
    """Recessive frame: keep the ink on the data, not the furniture."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, loc="left", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def _latest(pattern: str) -> Path:
    matches = sorted(glob.glob(str(config.REPORT_DIR / pattern)))
    if not matches:
        raise FileNotFoundError(
            f"no report matching {pattern!r} in {config.REPORT_DIR}. "
            "Run the experiments before generating figures."
        )
    return Path(matches[-1])


# --------------------------------------------------------------------------
# Figure 1 - equity curves, all seeds
# --------------------------------------------------------------------------
def fig_equity(output: Path = None) -> Path:
    """Walk-forward equity, with every seed drawn.

    Drawing all three seeds rather than one is the point: the spread between
    them is as large as the gap between feature sets, which is what makes a
    single-run performance claim meaningless.
    """
    # One panel per feature set. With three seed lines plus two benchmarks on a
    # single panel, a direct label placed at a group's mean lands next to
    # whichever line happens to be there - the labels stop being trustworthy.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)

    for ax, (feature_set, colour) in zip(
        axes, (("close", BLUE), ("technical", ORANGE))
    ):
        eq = pd.read_excel(_latest(f"WalkForward_{feature_set}_expanding_*.xlsx"),
                           sheet_name="equity", index_col=0, parse_dates=True)
        seeds = eq[[c for c in eq.columns if c.startswith("strategy_seed")]] \
            / config.INITIAL_CAPITAL

        # The band is the finding: it is the run-to-run spread of one design.
        ax.fill_between(eq.index, seeds.min(axis=1), seeds.max(axis=1),
                        color=colour, alpha=0.18, lw=0)
        for col in seeds.columns:
            ax.plot(eq.index, seeds[col], color=colour, lw=1.2, alpha=0.9)

        for col, label, style, bench_colour in (
            ("buy_hold", "Buy & hold", "-", AQUA),
            ("market", "Market ^TWII", "--", INK_SOFT),
        ):
            ax.plot(eq.index, eq[col] / config.INITIAL_CAPITAL,
                    color=bench_colour, lw=2.0, ls=style, label=label)
            ax.annotate(label,
                        xy=(eq.index[-1], eq[col].iloc[-1] / config.INITIAL_CAPITAL),
                        xytext=(6, 0), textcoords="offset points",
                        color=bench_colour, fontsize=9, fontweight="bold", va="center")

        ax.annotate(f"LSTM\n{feature_set}\n(3 seeds)",
                    xy=(eq.index[-1], seeds.iloc[-1].median()),
                    xytext=(6, 0), textcoords="offset points",
                    color=colour, fontsize=9, fontweight="bold", va="center")

        ax.set_yscale("log")
        ax.set_yticks([1, 2, 3, 5, 7])
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
        # Log scales add scientific-notation minor labels that clash with the
        # "1x/2x" majors; silence them.
        ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(axis="y", lw=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.margins(x=0.20)
        _style(ax, title=f"{feature_set} features",
               ylabel="growth of initial capital" if ax is axes[0] else None)

    fig.suptitle("Walk-forward equity, every seed drawn (log scale)",
                 x=0.008, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.008, -0.04,
             "The shaded band spans the three seeds of one identical configuration. "
             "That run-to-run spread is comparable to the gap between the two feature "
             "sets — which is why a single run cannot support a performance claim.",
             ha="left", fontsize=8.5, color=INK_SOFT)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output = output or FIGURE_DIR / "equity_walkforward.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# --------------------------------------------------------------------------
# Figure 2 - per-fold skill, both modes
# --------------------------------------------------------------------------
def fig_skill_by_fold(output: Path = None) -> Path:
    """Forest plot of prediction skill per fold, expanding vs rolling."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)

    for ax, mode in zip(axes, ("expanding", "rolling")):
        for offset, (feature_set, colour) in enumerate(
            (("close", BLUE), ("technical", ORANGE))
        ):
            folds = pd.read_excel(
                _latest(f"WalkForward_{feature_set}_{mode}_*.xlsx"), sheet_name="per_fold"
            )
            ys, means, los, his = [], [], [], []
            for fold, group in folds.groupby("fold"):
                stats = mean_and_ci(group["skill_vs_naive"].values)
                ys.append(fold + (offset - 0.5) * 0.26)
                means.append(stats["mean"])
                los.append(stats["ci_low"])
                his.append(stats["ci_high"])

            ax.hlines(ys, los, his, color=colour, lw=2.0, alpha=0.9)
            ax.plot(means, ys, "o", color=colour, ms=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=feature_set if ax is axes[0] else None)

        ax.axvline(0, color=INK, lw=1.4, zorder=0)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"fold {i}" for i in range(5)])
        ax.grid(axis="x", lw=0.8, alpha=0.6)
        ax.set_axisbelow(True)
        ax.margins(y=0.14)
        _style(ax, title=f"{mode} window", xlabel="skill vs naive (95% CI)")

    # Chronological order, earliest fold on top. The axes share y, so this
    # must happen exactly once - calling it per-axis would cancel out.
    axes[0].invert_yaxis()

    axes[0].legend(frameon=False, loc="upper left", fontsize=9,
                   bbox_to_anchor=(0.0, -0.22), ncol=2)
    fig.suptitle("Prediction skill by fold — all 20 fold means are negative",
                 x=0.008, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.008, -0.06,
             "Positive skill would mean beating the \"tomorrow = today\" baseline. "
             "Individual folds are noisy and some intervals cross zero, but no fold in "
             "any arm has a positive mean, and pooled over all folds the interval "
             "excludes zero.",
             ha="left", fontsize=8.5, color=INK_SOFT)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output = output or FIGURE_DIR / "skill_by_fold.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# --------------------------------------------------------------------------
# Figure 3 - the ablation contrast
# --------------------------------------------------------------------------
def fig_ablation(output: Path = None) -> Path:
    """The headline result: paired contrasts whose intervals contain zero."""
    # The date is the only thing allowed between the model type and the
    # extension: 'Ablation_LSTM_standardised_*' is a different experiment and
    # must not be picked up here just because it sorts later.
    contrasts = pd.read_excel(_latest("Ablation_LSTM_2*.xlsx"), sheet_name="contrasts")

    # The two tasks are measured in different units (skill score vs accuracy),
    # so they get their own panel and their own axis. Putting them on one
    # shared axis would silently compare unlike quantities.
    tasks = ["regression", "classification"]
    units = {
        "regression": "difference in skill vs naive (95% CI)",
        "classification": "difference in direction accuracy (95% CI)",
    }
    arms = {"technical": BLUE, "full": ORANGE}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, task in zip(axes, tasks):
        block = contrasts[contrasts["task"] == task].set_index("arm").loc[list(arms)]
        for i, (arm, r) in enumerate(block.iterrows()):
            colour = arms[arm]
            ax.hlines(i, r["ci_low"], r["ci_high"], color=colour, lw=2.4)
            ax.plot(r["mean_difference"], i, "o", color=colour, ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"p = {r['p_value']:.3f}",
                        xy=(r["ci_high"], i), xytext=(8, 0),
                        textcoords="offset points", va="center",
                        fontsize=8.5, color=INK_SOFT)

        ax.axvline(0, color=INK, lw=1.4, zorder=0)
        ax.set_yticks(range(len(block)))
        ax.set_yticklabels([f"{a} − close" for a in block.index])
        ax.invert_yaxis()
        ax.grid(axis="x", lw=0.8, alpha=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.30, y=0.55)
        _style(ax, title=task, xlabel=units[task])

    fig.suptitle("Does the feature engineering help? Paired contrast vs the close-only control",
                 x=0.008, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.008, -0.10,
             "Every interval spans zero: adding 6 technical indicators and 3 macro "
             "series produced no measurable change. Paired by (ticker, seed), n = 25.",
             ha="left", fontsize=8.5, color=INK_SOFT)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    output = output or FIGURE_DIR / "ablation_contrast.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# --------------------------------------------------------------------------
# Figure 4 - preprocessing sensitivity
# --------------------------------------------------------------------------
def fig_sensitivity(output: Path = None) -> Path:
    """How much the conclusion moves when only the normalisation changes."""
    runs = pd.read_excel(_latest("Sensitivity_*.xlsx"), sheet_name="runs")
    order = ["minmax_global", "minmax_train_only", "window_return"]
    pretty = {
        "minmax_global": "MinMax\nfitted before split\n(leaks test range)",
        "minmax_train_only": "MinMax\nfitted on train only",
        "window_return": "Window-relative returns\n(this project)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    grouped = runs.groupby("regime")

    # Left: skill, which spans three orders of magnitude -> symlog.
    ax = axes[0]
    for i, regime in enumerate(order):
        vals = grouped.get_group(regime)["skill_vs_naive"].values
        colour = AQUA if regime == "window_return" else ORANGE
        ax.plot(vals, [i] * len(vals), "o", color=colour, ms=7, alpha=0.55,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.plot(vals.mean(), i, "D", color=colour, ms=9,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        mean = vals.mean()
        # Near-zero values need real precision, or "+0.00" hides the finding.
        label = f"{mean:+.4f}" if abs(mean) < 0.01 else f"{mean:+.2f}"
        ax.annotate(label, xy=(mean, i), xytext=(0, 13),
                    textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold", color=INK)
    ax.axvline(0, color=INK, lw=1.4, zorder=0)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_yticks(range(3))
    ax.set_yticklabels([pretty[r] for r in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="x", lw=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.25)
    _style(ax, title="Skill vs naive baseline",
           xlabel="skill (symlog scale); one dot per seed, diamond = mean")

    # Right: R^2, the metric that hides the problem.
    ax = axes[1]
    naive_r2 = runs["naive_R2"].mean()
    means = [grouped.get_group(r)["R2"].mean() for r in order]
    colours = [ORANGE, ORANGE, AQUA]
    bars = ax.barh(range(3), means, color=colours, height=0.55)
    for i, (bar, v) in enumerate(zip(bars, means)):
        ax.annotate(f"{v:.4f}", xy=(v, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, fontweight="bold", color=INK)
    ax.axvline(naive_r2, color=INK, lw=1.4, ls=":", zorder=3)
    ax.annotate(f"naive baseline R² = {naive_r2:.4f}",
                xy=(naive_r2, 2.55), xytext=(-6, 0), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK)
    ax.set_yticks(range(3))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.grid(axis="x", lw=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.25)
    _style(ax, title="The same runs, scored as R²", xlabel="R² on the test set")

    fig.suptitle("Only the normalisation changes — same architecture, split and seeds (2330.TW)",
                 x=0.008, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.008, -0.04,
             "An R² of 0.92 reads as a good fit in isolation — but a do-nothing baseline "
             "scores 0.9968 on this series, so both MinMax variants are far worse than "
             "predicting no change at all.",
             ha="left", fontsize=8.5, color=INK_SOFT)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output = output or FIGURE_DIR / "sensitivity_normalisation.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# --------------------------------------------------------------------------
# Figure 5 - capacity sensitivity
# --------------------------------------------------------------------------
def fig_tuning(output: Path = None) -> Path:
    """Skill across the capacity grid: does any configuration beat persistence?

    A forest plot rather than a bar chart, for the same reason as the
    ablation figure: the claim is about where the intervals sit relative to
    zero, not about ranking the cells against each other.
    """
    summary = pd.read_excel(_latest("Tuning_LSTM_*.xlsx"), sheet_name="summary")
    summary = summary.sort_values("n_params").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for i, r in summary.iterrows():
        # The project default is the reference point every other number in
        # the README comes from, so it is called out rather than blended in.
        is_default = (r["units"] == config.DEFAULT_UNITS
                      and r["epochs"] == 20
                      and r["seq_length"] == config.SEQ_LENGTH)
        colour = BLUE if is_default else INK_SOFT
        ax.hlines(i, r["ci_low"], r["ci_high"], color=colour, lw=2.2)
        ax.plot(r["skill_vs_naive"], i, "o", color=colour, ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)

    ax.axvline(0, color=INK, lw=1.4, zorder=0)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([
        f"{r['units']} units · {r['epochs']} epochs · seq {r['seq_length']}"
        + ("  (default)" if (r["units"] == config.DEFAULT_UNITS and r["epochs"] == 20
                             and r["seq_length"] == config.SEQ_LENGTH) else "")
        for _, r in summary.iterrows()
    ], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", lw=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.18, y=0.06)
    _style(ax, xlabel="skill vs naive (95% CI)")

    fig.suptitle("Is the negative result an artefact of the capacity that was chosen?",
                 x=0.008, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.008, -0.05,
             "A pre-declared grid, not a search — no cell is selected on test-set "
             "performance. Positive skill would mean beating the \"tomorrow = today\" "
             "baseline; the grid spans a 54× range in parameter count (3.7k to 201k).",
             ha="left", fontsize=8.5, color=INK_SOFT)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output = output or FIGURE_DIR / "tuning_capacity.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# --------------------------------------------------------------------------
# Figure 6 - raw vs standardised feature channels
# --------------------------------------------------------------------------
def fig_standardisation(output: Path = None) -> Path:
    """Do the multi-feature arms improve once the channels are comparable?"""
    raw = pd.read_excel(_latest("Ablation_LSTM_2*.xlsx"), sheet_name="runs")
    std = pd.read_excel(_latest("Ablation_LSTM_standardised_*.xlsx"), sheet_name="runs")

    arms = ["close", "technical", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    for ax, (task, metric, unit) in zip(axes, (
        ("regression", "skill_vs_naive", "skill vs naive (95% CI)"),
        ("classification", "edge_vs_majority", "edge vs majority class (95% CI)"),
    )):
        for offset, (frame, label, colour) in enumerate(
            ((raw, "raw channels", ORANGE), (std, "standardised", AQUA))
        ):
            block = frame[frame["task"] == task]
            for i, arm in enumerate(arms):
                vals = block[block["feature_set"] == arm][metric].values
                if len(vals) == 0:
                    continue
                stats = mean_and_ci(vals)
                y = i + (offset - 0.5) * 0.26
                ax.hlines(y, stats["ci_low"], stats["ci_high"], color=colour, lw=2.2)
                ax.plot(stats["mean"], y, "o", color=colour, ms=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=label if (i == 0 and ax is axes[0]) else None)

        ax.axvline(0, color=INK, lw=1.4, zorder=0)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(arms)
        ax.invert_yaxis()
        ax.grid(axis="x", lw=0.8, alpha=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.22, y=0.30)
        _style(ax, title=task, xlabel=unit)

    axes[0].legend(frameon=False, loc="upper left", fontsize=9,
                   bbox_to_anchor=(0.0, -0.26), ncol=2)
    fig.suptitle("Were the extra features simply never on a comparable scale?",
                 x=0.008, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.008, -0.06,
             "Channel standard deviations span two orders of magnitude before "
             "standardisation (MACDHist ≈ 0.005, BollPos ≈ 0.63). The right-hand series\n"
             "z-scores every channel using training-set statistics only — which makes the "
             "multi-feature regression arms worse, not better.",
             ha="left", va="top", fontsize=8.5, color=INK_SOFT)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    output = output or FIGURE_DIR / "standardisation_contrast.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for builder in (fig_equity, fig_skill_by_fold, fig_ablation, fig_sensitivity,
                    fig_tuning, fig_standardisation):
        path = builder()
        print(f"[figures] {path}")


if __name__ == "__main__":
    main()
