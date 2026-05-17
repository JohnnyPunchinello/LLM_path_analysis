"""
plot_synergy_gap.py
-------------------
Plot Synergy Gap (H_ana - H_emp) vs Task Complexity for each model family.

Usage:
    python plot_synergy_gap.py                        # use embedded sample data
    python plot_synergy_gap.py --csv results/path_metrics.csv
    python plot_synergy_gap.py --csv results/path_metrics.csv --output figure2.png
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.15)
    SEABORN = True
except ImportError:
    warnings.warn("seaborn not installed — falling back to plain matplotlib style")
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.4,
        "font.size": 12,
    })
    SEABORN = False

from scipy import stats  # for pearsonr


# ---------------------------------------------------------------------------
# Complexity mapping
# ---------------------------------------------------------------------------
COMPLEXITY_MAP: dict[str, int] = {
    # Low  (0)
    "sst2":       0,
    "sst-2":      0,
    "piqa":       0,
    # Medium  (1)
    "boolq":      1,
    "arc_easy":   1,
    "arc-easy":   1,
    "rte":        1,
    "winogrande": 1,
    # High  (2)
    "hellaswag":  2,
    "arc_challenge": 2,
    "arc-challenge": 2,
    "gsm8k":      2,
    "humaneval":  2,
}

COMPLEXITY_LABELS = {0: "Low", 1: "Medium", 2: "High"}

# Canonical display names for tasks
TASK_DISPLAY: dict[str, str] = {
    "sst2": "SST-2", "sst-2": "SST-2",
    "piqa": "PIQA",
    "boolq": "BoolQ",
    "arc_easy": "ARC-Easy", "arc-easy": "ARC-Easy",
    "rte": "RTE",
    "winogrande": "WinoGrande",
    "hellaswag": "HellaSwag",
    "arc_challenge": "ARC-Challenge", "arc-challenge": "ARC-Challenge",
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
}


# ---------------------------------------------------------------------------
# Sample / fallback data
# (Replace with real CSV; values are representative of typical runs)
# ---------------------------------------------------------------------------
SAMPLE_DATA = {
    # Each row: (model, family, n_params_M, task, analytical_entropy, empirical_entropy)
    # Pythia family  (sequential, 6-layer → 32-layer depending on size)
    # H_ana grows with depth; H_emp tracks how much the task prunes the space
    "pythia": [
        # (model, n_params_M, task, H_ana, H_emp)
        # --- 70M (6 layers, max_path_len=12) ---
        ("pythia-70m",   70,   "sst2",          6.58, 5.82),
        ("pythia-70m",   70,   "boolq",         6.58, 5.95),
        ("pythia-70m",   70,   "arc_easy",      6.58, 6.01),
        ("pythia-70m",   70,   "hellaswag",     6.58, 6.21),
        # --- 160M (12 layers) ---
        ("pythia-160m",  160,  "sst2",          7.58, 6.51),
        ("pythia-160m",  160,  "boolq",         7.58, 6.72),
        ("pythia-160m",  160,  "arc_easy",      7.58, 6.89),
        ("pythia-160m",  160,  "hellaswag",     7.58, 7.12),
        # --- 410M (24 layers) ---
        ("pythia-410m",  410,  "sst2",          8.58, 7.12),
        ("pythia-410m",  410,  "boolq",         8.58, 7.43),
        ("pythia-410m",  410,  "arc_easy",      8.58, 7.68),
        ("pythia-410m",  410,  "hellaswag",     8.58, 8.01),
        # --- 1B (16 layers) ---
        ("pythia-1b",    1000, "sst2",          8.00, 6.52),
        ("pythia-1b",    1000, "boolq",         8.00, 6.81),
        ("pythia-1b",    1000, "arc_easy",      8.00, 7.05),
        ("pythia-1b",    1000, "hellaswag",     8.00, 7.43),
        # --- 1.4B (24 layers) ---
        ("pythia-1.4b",  1400, "sst2",          8.58, 6.85),
        ("pythia-1.4b",  1400, "boolq",         8.58, 7.21),
        ("pythia-1.4b",  1400, "arc_easy",      8.58, 7.48),
        ("pythia-1.4b",  1400, "hellaswag",     8.58, 7.89),
        # --- 2.8B (32 layers) ---
        ("pythia-2.8b",  2800, "sst2",          9.17, 7.15),
        ("pythia-2.8b",  2800, "boolq",         9.17, 7.58),
        ("pythia-2.8b",  2800, "arc_easy",      9.17, 7.82),
        ("pythia-2.8b",  2800, "hellaswag",     9.17, 8.31),
    ],
    # GPT-2 family  (sequential)
    "gpt2": [
        # (model, n_params_M, task, H_ana, H_emp)
        # --- gpt2 (12 layers) ---
        ("gpt2",         117,  "sst2",          7.58, 6.42),
        ("gpt2",         117,  "boolq",         7.58, 6.71),
        ("gpt2",         117,  "arc_easy",      7.58, 6.95),
        ("gpt2",         117,  "hellaswag",     7.58, 7.18),
        # --- gpt2-medium (24 layers) ---
        ("gpt2-medium",  345,  "sst2",          8.58, 7.05),
        ("gpt2-medium",  345,  "boolq",         8.58, 7.38),
        ("gpt2-medium",  345,  "arc_easy",      8.58, 7.61),
        ("gpt2-medium",  345,  "hellaswag",     8.58, 7.91),
        # --- gpt2-large (36 layers) ---
        ("gpt2-large",   774,  "sst2",          9.17, 7.35),
        ("gpt2-large",   774,  "boolq",         9.17, 7.72),
        ("gpt2-large",   774,  "arc_easy",      9.17, 7.98),
        ("gpt2-large",   774,  "hellaswag",     9.17, 8.39),
        # --- gpt2-xl (48 layers) ---
        ("gpt2-xl",      1558, "sst2",          9.58, 7.61),
        ("gpt2-xl",      1558, "boolq",         9.58, 8.02),
        ("gpt2-xl",      1558, "arc_easy",      9.58, 8.29),
        ("gpt2-xl",      1558, "hellaswag",     9.58, 8.74),
    ],
}


def load_sample_data() -> pd.DataFrame:
    rows = []
    for family, records in SAMPLE_DATA.items():
        for model, n_params_M, task, h_ana, h_emp in records:
            rows.append({
                "model":               model,
                "family":              family,
                "n_params_M":          n_params_M,
                "task":                task,
                "analytical_entropy":  h_ana,
                "empirical_entropy":   h_emp,
                "synergy_gap":         h_ana - h_emp,
            })
    df = pd.DataFrame(rows)
    df["task_complexity"] = df["task"].str.lower().map(COMPLEXITY_MAP)
    df["task_display"] = df["task"].str.lower().map(TASK_DISPLAY).fillna(df["task"])
    return df


def load_csv_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names (CSV uses 'model', 'task', 'synergy_gap',
    # 'analytical_entropy', 'empirical_entropy' per the CSV schema in README)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"model", "task", "synergy_gap", "analytical_entropy", "empirical_entropy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Infer family from model name
    def infer_family(model_id: str) -> str:
        m = model_id.lower()
        if "pythia" in m:
            return "pythia"
        if "gpt2" in m or "gpt-2" in m:
            return "gpt2"
        if "gpt-j" in m or "gptj" in m:
            return "gptj"
        if "llama" in m:
            return "llama"
        if "falcon" in m:
            return "falcon"
        if "neo" in m:
            return "neo"
        return "other"

    df["family"] = df["model"].apply(infer_family)

    # Map complexity
    df["task_complexity"] = df["task"].str.lower().map(COMPLEXITY_MAP)
    unmapped = df[df["task_complexity"].isna()]["task"].unique()
    if len(unmapped) > 0:
        warnings.warn(f"Unknown tasks (will be dropped): {unmapped.tolist()}")
    df = df.dropna(subset=["task_complexity"])
    df["task_complexity"] = df["task_complexity"].astype(int)

    df["task_display"] = df["task"].str.lower().map(TASK_DISPLAY).fillna(df["task"])

    # Average over samples if multiple rows per (model, task)
    group_cols = ["model", "family", "task", "task_complexity", "task_display",
                  "analytical_entropy"]
    df = (df.groupby(group_cols, as_index=False)
            .agg(synergy_gap=("synergy_gap", "mean"),
                 empirical_entropy=("empirical_entropy", "mean")))

    return df


# ---------------------------------------------------------------------------
# Pearson analysis
# ---------------------------------------------------------------------------
def analyse_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (family, model), compute Pearson r between task_complexity and synergy_gap.
    Also classify the trend as 'Closing Gap', 'Fixed Gap', or 'Widening Gap'.
    """
    records = []
    for (family, model), grp in df.groupby(["family", "model"]):
        grp = grp.sort_values("task_complexity")
        if len(grp) < 2:
            continue
        x = grp["task_complexity"].values.astype(float)
        y = grp["synergy_gap"].values.astype(float)
        r, p = stats.pearsonr(x, y)
        # Trend classification
        if r < -0.3 and p < 0.1:
            trend = "Closing Gap  (↓ with complexity)"
        elif r > 0.3 and p < 0.1:
            trend = "Widening Gap (↑ with complexity)"
        else:
            trend = "Fixed Gap    (≈ constant)"
        records.append({
            "family": family,
            "model":  model,
            "pearson_r": round(r, 3),
            "p_value":   round(p, 4),
            "n_tasks":   len(grp),
            "trend":     trend,
        })
    return pd.DataFrame(records).sort_values(["family", "model"])


def print_analysis(corr_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  PEARSON CORRELATION: Task Complexity → Synergy Gap")
    print("=" * 70)
    for family, grp in corr_df.groupby("family"):
        print(f"\n  [{family.upper()} family]")
        for _, row in grp.iterrows():
            print(f"    {row['model']:<22}  r = {row['pearson_r']:+.3f}  "
                  f"p = {row['p_value']:.4f}  n = {row['n_tasks']}  →  {row['trend']}")

    print("\n" + "-" * 70)
    print("  TREND SUMMARY")
    print("-" * 70)
    closing  = corr_df[corr_df["trend"].str.startswith("Closing")]
    fixed    = corr_df[corr_df["trend"].str.startswith("Fixed")]
    widening = corr_df[corr_df["trend"].str.startswith("Widening")]
    if not closing.empty:
        print(f"\n  Closing Gap ({len(closing)} models):")
        for _, r in closing.iterrows():
            print(f"    • {r['model']}  (r = {r['pearson_r']:+.3f})")
    if not fixed.empty:
        print(f"\n  Fixed Gap ({len(fixed)} models):")
        for _, r in fixed.iterrows():
            print(f"    • {r['model']}  (r = {r['pearson_r']:+.3f})")
    if not widening.empty:
        print(f"\n  Widening Gap ({len(widening)} models):")
        for _, r in widening.iterrows():
            print(f"    • {r['model']}  (r = {r['pearson_r']:+.3f})")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]
JITTER_STRENGTH = 0.04   # horizontal jitter to separate overlapping points


def _family_plot(ax: plt.Axes, family_df: pd.DataFrame, family: str,
                 corr_df: pd.DataFrame, is_sample: bool) -> None:
    """Draw one subplot for a single model family."""
    models = sorted(family_df["model"].unique(),
                    key=lambda m: family_df.loc[family_df["model"] == m,
                                                "analytical_entropy"].iloc[0])

    palette = (sns.color_palette("tab10", len(models))
               if SEABORN else plt.cm.tab10(np.linspace(0, 0.9, len(models))))

    rng = np.random.default_rng(42)

    for i, model in enumerate(models):
        mdf = family_df[family_df["model"] == model].sort_values("task_complexity")
        x = mdf["task_complexity"].values.astype(float)
        y = mdf["synergy_gap"].values.astype(float)

        jitter = rng.uniform(-JITTER_STRENGTH, JITTER_STRENGTH, size=len(x))
        color  = palette[i]
        marker = MARKERS[i % len(MARKERS)]

        # --- line ---
        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8, zorder=2)

        # --- scatter ---
        ax.scatter(x + jitter, y, color=color, marker=marker, s=70,
                   label=model, zorder=3, linewidths=0.5, edgecolors="white")

        # --- Pearson annotation at the right end ---
        row = corr_df[(corr_df["family"] == family) & (corr_df["model"] == model)]
        if not row.empty:
            r_val = row["pearson_r"].iloc[0]
            ax.annotate(f"r={r_val:+.2f}",
                        xy=(x[-1], y[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=7.5, color=color, va="center")

    # y=0 reference
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--",
               alpha=0.55, zorder=1, label="Full utilisation (Δ=0)")

    # axes labels & title
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_xlim(-0.35, 2.75)
    ax.set_xlabel("Task Complexity", fontsize=11)
    ax.set_ylabel(r"Synergy Gap  $\Delta H = H_{\mathrm{ana}} - H_{\mathrm{emp}}$  (bits)",
                  fontsize=10)
    title = f"{family.upper()} Family"
    if is_sample:
        title += "  [sample data]"
    ax.set_title(title, fontsize=12, fontweight="bold")

    # legend — place outside to avoid overlap
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85,
              borderpad=0.6, handlelength=1.5)


def plot_figure(df: pd.DataFrame, corr_df: pd.DataFrame,
                families: list[str], is_sample: bool,
                output_path: str) -> None:
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.2),
                             sharey=False, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        fdf = df[df["family"] == family]
        _family_plot(ax, fdf, family, corr_df, is_sample)

    # Super-title
    suptitle = "Synergy Gap vs Task Complexity"
    if is_sample:
        suptitle += " (illustrative sample data - replace with real CSV)"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved → {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv",     default=None,
                        help="Path to path_metrics.csv from experiment_runner.py")
    parser.add_argument("--output",  default="results/synergy_gap_vs_complexity.png",
                        help="Output figure path (default: results/synergy_gap_vs_complexity.png)")
    parser.add_argument("--families", default="pythia,gpt2",
                        help="Comma-separated model families to plot (default: pythia,gpt2)")
    args = parser.parse_args()

    # --- Load data ---
    is_sample = False
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            sys.exit(f"ERROR: CSV not found at {csv_path}")
        print(f"Loading data from {csv_path} …")
        df = load_csv_data(str(csv_path))
    else:
        print("No --csv provided — using embedded sample data for illustration.")
        print("Run: python plot_synergy_gap.py --csv results/path_metrics.csv\n")
        df = load_sample_data()
        is_sample = True

    # --- Filter to requested families ---
    families = [f.strip().lower() for f in args.families.split(",")]
    available = df["family"].unique().tolist()
    families = [f for f in families if f in available]
    if not families:
        sys.exit(f"ERROR: None of the requested families {families} found in data.\n"
                 f"Available families: {available}")

    df_plot = df[df["family"].isin(families)].copy()

    # --- Correlation analysis ---
    corr_df = analyse_correlations(df_plot)
    print_analysis(corr_df)

    # --- Plot ---
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plot_figure(df_plot, corr_df, families, is_sample, output_path)


if __name__ == "__main__":
    main()
