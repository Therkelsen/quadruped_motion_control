"""Create a set of diagnostic figures from CSV summary data.

This script reads CSV files from a data directory (default
`data_analysis/data`) and writes a set of figures into
`data_analysis/figures/` (boxplots, ecdf, histograms, violin, qq and
per-model time series and scatter plots).

Example:
  python3 data_analysis/data_analysis.py --data data_analysis/data --exclude episode

The --exclude argument accepts a comma-separated list of column names to
skip when creating component-wise plots.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def find_csvs(data_dir: str) -> List[str]:
    pattern = os.path.join(data_dir, "*.csv")
    return sorted(glob.glob(pattern))


def model_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def load_data(data_dir: str) -> Dict[str, List[pd.DataFrame]]:
    files = find_csvs(data_dir)
    models: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    for p in files:
        try:
            df = pd.read_csv(p)
            # Ensure common columns exist
            models[model_name_from_path(p)].append(df)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    return models


def ensure_dirs(base_dir: str):
    subdirs = [
        "figures/boxplots",
        "figures/ecdf_plots",
        "figures/histograms/combined",
        "figures/qq_plots",
        "figures/scatter_plots",
        "figures/violin_plots",
        "figures/line_plots",
    ]
    for s in subdirs:
        os.makedirs(os.path.join(base_dir, s), exist_ok=True)


def concat_runs(runs: List[pd.DataFrame]) -> pd.DataFrame:
    # Concatenate runs vertically and ignore index to treat episodes as samples
    if len(runs) == 0:
        return pd.DataFrame()
    return pd.concat(runs, ignore_index=True)


def plot_boxplot(models: Dict[str, pd.DataFrame], out_base: str, column: str):
    plt.figure(figsize=(6, 4))
    data = [df[column].dropna() for df in models.values()]
    labels = list(models.keys())
    sns.boxplot(data=data)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel(column)
    plt.title(f"{column} — boxplot by model")
    plt.tight_layout()
    plt.savefig(out_base + f"/boxplots/{column}_boxplot.png", dpi=200)
    plt.close()


def plot_ecdf(models: Dict[str, pd.DataFrame], out_base: str, column: str):
    plt.figure(figsize=(6, 4))
    for name, df in models.items():
        if column in df.columns:
            sns.ecdfplot(df[column].dropna(), label=name)
    plt.xlabel(column)
    plt.ylabel("ECDF")
    plt.title(f"ECDF — {column}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_base + f"/ecdf_plots/{column}_ecdf.png", dpi=200)
    plt.close()


def plot_histogram_combined(models: Dict[str, pd.DataFrame], out_base: str, column: str):
    plt.figure(figsize=(6, 4))
    for name, df in models.items():
        if column in df.columns:
            sns.histplot(df[column].dropna(), kde=False, stat='density', label=name, alpha=0.4, bins=40)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.title(f"Histogram (combined) — {column}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_base + f"/histograms/combined/{column}_hist_combined.png", dpi=200)
    plt.close()


def plot_violin(models: Dict[str, pd.DataFrame], out_base: str, column: str):
    plt.figure(figsize=(6, 4))
    df_list = []
    for name, df in models.items():
        if column in df.columns:
            tmp = pd.DataFrame({"model": name, column: df[column].dropna()})
            df_list.append(tmp)
    if not df_list:
        return
    all_df = pd.concat(df_list, ignore_index=True)
    sns.violinplot(x="model", y=column, data=all_df)
    plt.title(f"Violin — {column}")
    plt.tight_layout()
    plt.savefig(out_base + f"/violin_plots/{column}_violin.png", dpi=200)
    plt.close()


def plot_qq(models: Dict[str, pd.DataFrame], out_base: str, column: str):
    try:
        import scipy.stats as stats
    except Exception:
        print("scipy not available — skipping QQ plots")
        return

    for name, df in models.items():
        if column not in df.columns:
            continue
        data = df[column].dropna().values
        if len(data) < 4:
            continue
        plt.figure(figsize=(5, 4))
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f"QQ plot ({name}) — {column}")
        plt.tight_layout()
        safe_name = name.replace('/', '_')
        plt.savefig(out_base + f"/qq_plots/{safe_name}_{column}_qq.png", dpi=200)
        plt.close()


def plot_scatter_per_model(models: Dict[str, pd.DataFrame], out_base: str, xcol: str, ycol: str):
    for name, df in models.items():
        if xcol not in df.columns or ycol not in df.columns:
            continue
        plt.figure(figsize=(5.5, 4))
        plt.scatter(df[xcol], df[ycol], s=12, alpha=0.6)
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.title(f"{name}: {ycol} vs {xcol}")
        plt.tight_layout()
        safe_name = name.replace('/', '_')
        out_dir = os.path.join(out_base, 'scatter_plots')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{safe_name}_{ycol}_vs_{xcol}.png"), dpi=200)
        plt.close()


def plot_line_per_model(models: Dict[str, pd.DataFrame], out_base: str, column: str):
    for name, df in models.items():
        if column not in df.columns:
            continue
        plt.figure(figsize=(7, 3))
        sns.lineplot(x=df.index, y=df[column].values)
        plt.xlabel('index')
        plt.ylabel(column)
        plt.title(f"{name} — {column} over samples")
        plt.tight_layout()
        safe_name = name.replace('/', '_')
        plt.savefig(out_base + f"/line_plots/{safe_name}_{column}_line.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data_analysis/data', help='Directory with CSV data files')
    parser.add_argument('--out', default='data_analysis', help='Base output folder (will create figures/ inside)')
    parser.add_argument('--exclude', default='', help='Comma-separated columns to exclude from component plots')
    args = parser.parse_args()

    base = args.out
    figures_base = os.path.join(base)
    ensure_dirs(base)

    raw = load_data(args.data)
    # concatenate runs per model into a single DataFrame for distribution plots
    models = {name: concat_runs(runs) for name, runs in raw.items()}

    if not models:
        print(f"No CSV data found in {args.data}")
        return

    # Decide which numeric columns to plot
    all_cols = set()
    for df in models.values():
        all_cols.update(df.columns.tolist())
    exclude = {c.strip() for c in args.exclude.split(',') if c.strip()}
    num_cols = [c for c in sorted(all_cols) if c not in exclude]

    # Prefer plotting 'total_reward' and 'length' first if present
    preferred = []
    if 'total_reward' in num_cols:
        preferred.append('total_reward')
    if 'length' in num_cols:
        preferred.append('length')

    others = [c for c in num_cols if c not in preferred]
    columns_to_plot = preferred + others

    # Create summary plots for a handful of columns
    for col in columns_to_plot:
        # Only plot numeric columns
        try:
            # quick check using first model that has the column
            sample_df = next(df for df in models.values() if col in df.columns)
        except StopIteration:
            continue
        # coerce to numeric where possible
        for k in models:
            if col in models[k].columns:
                models[k][col] = pd.to_numeric(models[k][col], errors='coerce')

        plot_boxplot(models, figures_base, col)
        plot_ecdf(models, figures_base, col)
        plot_histogram_combined(models, figures_base, col)
        plot_violin(models, figures_base, col)
        plot_qq(models, figures_base, col)
        # line plots per model
        plot_line_per_model(models, figures_base, col)

    # Scatter plots: pick pairs among the first few numeric columns
    scatter_cols = [c for c in columns_to_plot if c in ['tracking_lin_vel', 'tracking_ang_vel', 'total_reward', 'length']]
    if len(scatter_cols) >= 2:
        for i in range(len(scatter_cols)):
            for j in range(i + 1, len(scatter_cols)):
                plot_scatter_per_model(models, figures_base, scatter_cols[i], scatter_cols[j])

    print(f"Saved figures under {figures_base}/figures/")


if __name__ == '__main__':
    main()
