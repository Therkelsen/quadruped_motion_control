import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.multivariate.manova import MANOVA


def load_csv_file(file_path):
    """Load a single CSV and return (filename, DataFrame)."""
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path)
    return filename, df


def load_csv_folder(folder_path: str):
    """Load all CSV files in a folder into a dict of DataFrames using threads."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Provided path is not a directory: {folder_path}")

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV files found in the provided directory.")

    data = {}
    file_paths = [os.path.join(folder_path, f) for f in csv_files]

    n_proc = os.cpu_count() or 1  # fallback to 1 if detection fails
    max_workers = max(1, n_proc - 1)  # leave one core free

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(load_csv_file, fp): fp for fp in file_paths}
        for future in as_completed(future_to_file):
            filename, df = future.result()
            data[filename] = df

    return data


def select_columns(df: pd.DataFrame, exclude=None):
    """Select numeric columns with optional exclusions."""
    columns = df.select_dtypes(include='number').columns.tolist()
    exclude = exclude or []
    columns = [col for col in columns if col not in exclude]

    if not columns:
        print("No numeric columns found to plot.")
        return [], 0

    return columns, len(columns)


def create_output_dirs(paths):
    """Create multiple directories if they do not exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def calculate_grid(n_plots):
    """Calculate a square (or near-square) grid layout for n_plots."""
    n_rows = math.ceil(math.sqrt(n_plots))
    n_cols = math.ceil(n_plots / n_rows)
    return n_rows, n_cols


def save_figure(fig, filename, title=None):
    """Save a matplotlib figure to file with suptitle."""
    if title:
        plt.suptitle(title, fontsize=16)
    fig.savefig(filename)
    plt.close(fig)


def calculate_figsize(n_rows, n_cols):
    return (4*n_cols, 4*n_rows)


def turn_off_unused_axes(fig, axes, last_used_index):
    for j in range(last_used_index+1, len(axes)):
        fig.delaxes(axes[j])


def plot_grid_base(df, columns, plot_func, title, filename):
    """Generic grid plotting function for histograms, QQ-plots, scatter."""
    n_cols_data = len(columns)
    if n_cols_data == 0:
        return

    n_rows, n_cols_grid = calculate_grid(n_cols_data)
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=calculate_figsize(n_cols_grid, n_rows))
    fig.subplots_adjust(wspace=1, hspace=1)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        data = df[col]
        if len(data) == 0:
            print(f"Skipping column {col}: no valid data")
            continue
        plot_func(axes[i], df, col)

    turn_off_unused_axes(fig, axes, i)

    save_figure(fig, filename, title)


def histogram_grid(df, exclude=None, filename="", title="Histograms"):
    columns, _ = select_columns(df, exclude)
    n_rows, n_cols_grid = calculate_grid(len(columns))
    
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=calculate_figsize(n_cols_grid, n_rows))
    fig.subplots_adjust(wspace=1, hspace=1)
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        sns.histplot(df[col], bins=15, kde=False, color='skyblue', ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].grid(True, alpha=0.3)
        
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    save_figure(fig, filename, title)


def boxplot_grid(df, exclude=None, filename="", title="Boxplots"):
    columns, _ = select_columns(df, exclude)
    n_rows, n_cols_grid = calculate_grid(len(columns))

    fig, axes = plt.subplots(n_rows, n_cols_grid,
                             figsize=calculate_figsize(n_cols_grid, n_rows))
    fig.subplots_adjust(wspace=1, hspace=1)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Value")
        axes[i].grid(True, alpha=0.3)

    # turn off unused axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    save_figure(fig, filename, title)


def boxplot_across_files(data_dict, exclude=None, output_dir=""):
    """
    Creates one big figure with subplots, where each subplot is a boxplot
    of one numeric variable across CSV files.
    """
    # Collect all numeric columns across files
    all_columns = set()
    for df in data_dict.values():
        all_columns.update(df.select_dtypes(include='number').columns.tolist())

    if exclude:
        all_columns = [c for c in all_columns if c not in exclude]
    else:
        all_columns = list(all_columns)

    if not all_columns:
        return

    # Ensure output directory exists
    create_output_dirs([output_dir])

    n_vars = len(all_columns)
    n_rows, n_cols_grid = calculate_grid(n_vars)
    if n_cols_grid > 4:  # Cap width at 4 columns
        n_cols_grid = 4
        n_rows = math.ceil(n_vars / n_cols_grid)

    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=calculate_figsize(n_cols_grid, n_rows))
    axes = axes.flatten()

    for i, col in enumerate(all_columns):
        plot_data = []
        labels = []

        for filename, df in data_dict.items():
            if col in df.columns:
                plot_data.append(df[col].dropna().values)
                labels.append(filename.replace(".csv", ""))

        if not plot_data:
            continue

        sns.boxplot(data=plot_data, ax=axes[i])
        axes[i].set_xticks(range(len(labels)))
        axes[i].set_xticklabels(labels, rotation=30)  # less rotation
        axes[i].set_ylabel(col)
        axes[i].set_title(col)
        axes[i].grid(True, alpha=0.3)

    turn_off_unused_axes(fig, axes, i)

    # Increase space at the bottom for x-axis labels
    fig.subplots_adjust(wspace=0.4, hspace=1)

    out_file = os.path.join(output_dir, "boxplots_all_vars.png")
    save_figure(fig, out_file, "Boxplots of all variables across CSV files")


def qqplot_grid(df, exclude=None, filename="", title="QQ-Plots"):
    """
    Create a grid of QQ-plots for the given DataFrame with Seaborn styling.
    """
    columns, _ = select_columns(df, exclude)
    n_rows, n_cols_grid = calculate_grid(len(columns))
    
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=calculate_figsize(n_cols_grid, n_rows))
    fig.subplots_adjust(wspace=1, hspace=1)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        stats.probplot(df[col], dist="norm", plot=axes[i])
        axes[i].set_title(col, fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # Optional: adjust tick parameters for readability
        axes[i].tick_params(axis='both', which='major', labelsize=10)

    turn_off_unused_axes(fig, axes, i)

    save_figure(fig, filename, title)


def scatter_grid(df, exclude=None, output_dir="", title_prefix="Scatter Plots"):
    columns, _ = select_columns(df, exclude)

    for x_col in columns:
        y_cols = columns
        n_rows, n_cols_grid = calculate_grid(len(y_cols))
        fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=calculate_figsize(n_cols_grid, n_rows))
        fig.subplots_adjust(wspace=1, hspace=1)
        axes = axes.flatten()

        for i, y_col in enumerate(y_cols):
            sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.6, ax=axes[i])
            axes[i].set_xlabel(x_col)
            axes[i].set_ylabel(y_col)
            axes[i].set_title(f"{y_col} vs {x_col}")
            axes[i].grid(True, alpha=0.3)

        turn_off_unused_axes(fig, axes, i)

        out_file = os.path.join(output_dir, f"{x_col}_scatter.png")
        save_figure(fig, out_file, f"{title_prefix}: {x_col} vs All")


def run_manova(data_dict, exclude=None):
    """
    Correct MANOVA: combine all CSVs into one dataset,
    each CSV is a 'group' (RL algorithm), DVs = numeric columns.
    """
    # 1. Collect all numeric columns across files
    all_columns = set()
    for df in data_dict.values():
        all_columns.update(df.select_dtypes(include='number').columns.tolist())

    if exclude:
        all_columns = [c for c in all_columns if c not in exclude]
    else:
        all_columns = list(all_columns)

    if not all_columns:
        print("No numeric columns found for MANOVA.")
        return

    # 2. Combine CSVs into one DataFrame with a 'group' column
    combined_data = []
    for filename, df in data_dict.items():
        subset = df[all_columns].copy()
        subset['group'] = filename.replace(".csv", "")
        combined_data.append(subset)

    combined_df = pd.concat(combined_data, ignore_index=True)

    # 3. Build formula: DV1 + DV2 + ... + DVn ~ group
    formula = " + ".join(all_columns) + " ~ group"
    
    # 4. Run MANOVA
    manova = MANOVA.from_formula(formula, data=combined_df)
    print("\n=== MANOVA Results ===")
    print(manova.mv_test())


def run_anovas(data_dict, exclude=None):
    """
    Run one-way ANOVA for each numeric dependent variable across groups (CSV files).
    Prints results similar to MANOVA.

    Uses: statsmodels.api as sm, statsmodels.formula.api as smf
    """

    # 1. Collect all numeric column names across files
    all_columns = set()
    for df in data_dict.values():
        all_columns.update(df.select_dtypes(include='number').columns.tolist())

    if exclude:
        all_columns = [c for c in all_columns if c not in exclude]
    else:
        all_columns = list(all_columns)

    if not all_columns:
        print("No numeric columns available for ANOVA.")
        return

    # 2. Combine data into one dataframe with a 'group' column
    combined_rows = []
    for filename, df in data_dict.items():
        name = filename.replace(".csv", "")
        tmp = df[all_columns].copy()
        tmp["group"] = name
        combined_rows.append(tmp)

    combined_df = pd.concat(combined_rows, ignore_index=True)

    print("\n=== ANOVA Results (one-way) ===")

    # 3. Run ANOVA for each DV
    for dv in all_columns:
        formula = f"{dv} ~ C(group)"  # C() ensures categorical
        model = smf.ols(formula, data=combined_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print(f"\n--- ANOVA for dependent variable: {dv} ---")
        print(anova_table)
        print("-------------------------------------------")
    

def run_tukey_hsd(data_dict, exclude=None):
    """
    Run Tukey HSD (post-hoc) test for each numeric dependent variable across groups (CSV files).
    Prints results in terminal.
    """
    import statsmodels.api as sm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # 1. Collect all numeric column names across files
    all_columns = set()
    for df in data_dict.values():
        all_columns.update(df.select_dtypes(include='number').columns.tolist())

    if exclude:
        all_columns = [c for c in all_columns if c not in exclude]
    else:
        all_columns = list(all_columns)

    if not all_columns:
        print("No numeric columns available for Tukey HSD.")
        return

    # 2. Combine data into one dataframe with a 'group' column
    combined_rows = []
    for filename, df in data_dict.items():
        name = filename.replace(".csv", "")
        tmp = df[all_columns].copy()
        tmp["group"] = name
        combined_rows.append(tmp)

    combined_df = pd.concat(combined_rows, ignore_index=True)

    print("\n=== Tukey HSD Post-hoc Test ===")

    # 3. Run Tukey HSD for each dependent variable
    for dv in all_columns:
        try:
            tukey = pairwise_tukeyhsd(endog=combined_df[dv],
                                    groups=combined_df["group"],
                                    alpha=0.05)
            print(f"\n--- Tukey HSD for dependent variable: {dv} ---")
            print(tukey)
            print("-------------------------------------------")
        except Exception as e:
            print(f"Could not run Tukey HSD for {dv}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Load evaluation CSV data and create plots.")
    parser.add_argument("--data", type=str, required=True, help="Path to the folder containing CSV files.")
    parser.add_argument("--exclude", type=str, default="", help="Comma-separated list of column names to exclude.")

    args = parser.parse_args()
    exclude = [c.strip() for c in args.exclude.split(",")] if args.exclude else []

    print(f"Loading data from: {args.data}")
    data = load_csv_folder(args.data)
    print(f"Loaded {len(data)} CSV files:")

    base_out_path = "data_analysis/figures/"
    hist_out_path = f"{base_out_path}histograms/"
    qq_out_path   = f"{base_out_path}qq_plots/"
    box_out_path  = f"{base_out_path}boxplots/"
    scatter_out_path = f"{base_out_path}scatter_plots/"

    # Create directories
    create_output_dirs([hist_out_path, qq_out_path, box_out_path, scatter_out_path])

    sns.set(style="whitegrid", palette="pastel", context="notebook")

    for filename, df in data.items():
        print(f" - {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        base_name = filename.replace(".csv", "")

        # Histograms
        histogram_grid(
            df, exclude=exclude,
            filename=f"{hist_out_path}{base_name}_histgrid.png",
            title=f"Histograms: {filename}"
        )

        # QQ plots
        qqplot_grid(
            df, exclude=exclude,
            filename=f"{qq_out_path}{base_name}_qqgrid.png",
            title=f"QQ-Plots: {filename}"
        )

        # Boxplots
        boxplot_grid(
            df, exclude=exclude,
            filename=f"{box_out_path}{base_name}_boxgrid.png",
            title=f"Boxplots: {filename}"
        )

        # Scatter plots (folder per CSV)
        scatter_dir = f"{scatter_out_path}{base_name}/"
        create_output_dirs([scatter_dir])

        scatter_grid(
            df, exclude=exclude,
            output_dir=scatter_dir,
            title_prefix=f"Scatter Plots: {filename}"
        )

    # Boxplots across files
    boxplot_across_files(
        data,
        exclude=exclude,
        output_dir=box_out_path
    )

    # MANOVA across CSV files — only once
    print("\nRunning MANOVA across CSV files:")
    run_manova(data, exclude=exclude)

    print("\nRunning ANOVAs across CSV files:")
    run_anovas(data, exclude=exclude)

    print("\nRunning Tukey HSD post-hoc tests across CSV files:")
    run_tukey_hsd(data, exclude=exclude)


if __name__ == "__main__":
    main()
