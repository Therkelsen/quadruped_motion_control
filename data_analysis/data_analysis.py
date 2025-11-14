import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def load_csv_folder(folder_path: str):
    """Load all CSV files in a folder into a dict of dataframes."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Provided path is not a directory: {folder_path}")

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV files found in the provided directory.")

    data = {}
    for file in csv_files:
        full_path = os.path.join(folder_path, file)
        df = pd.read_csv(full_path)
        data[file] = df

    return data


def select_columns(df: pd.DataFrame, exclude=None):
    # Select numeric columns by default
    columns = df.select_dtypes(include='number').columns.tolist()

    if exclude is None:
        exclude = []

    # Apply exclusion
    columns = [col for col in columns if col not in exclude]

    n_cols_data = len(columns)
    if n_cols_data == 0:
        print("No numeric columns found to plot.")
        return
    
    return columns, n_cols_data

def create_output_dirs(paths):
    """
    Create multiple directories if they do not exist.

    Parameters:
        paths (list of str): List of directory paths to create.
    """
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Ensured directory exists: {path}")

def histogram_grid(df: pd.DataFrame, 
                   exclude=None,
                   filename="data_analysis/figures/histograms.png", 
                   title="Histograms"):
    """
    Create a grid of histograms for the given DataFrame.

    Parameters:
        df (pd.DataFrame): The dataframe containing data.
        exclude (list or None): Columns to exclude from plotting.
        filename (str): Path to save the figure.
        title (str): Figure title.
    """
    columns, n_cols_data = select_columns(df, exclude)

    # Prefer square layout
    n_rows = math.ceil(math.sqrt(n_cols_data))
    n_cols_grid = math.ceil(n_cols_data / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(4*n_cols_grid, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        data = df[col].dropna()
        if len(data) == 0:
            print(f"Skipping column {col}: no valid data")
            continue

        # Plot histogram as counts (default)
        axes[i].hist(data, bins=15, alpha=0.6, color='skyblue', edgecolor='black')

        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].grid(True, alpha=0.3)

    # Turn off unused axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.close()
    print(f"Saved histogram grid to {filename}")


def qqplot_grid(df: pd.DataFrame, 
                exclude=None,
                filename="data_analysis/figures/qqplots.png", 
                title="QQ-Plots"):
    """
    Create a grid of QQ-plots for the given DataFrame.

    Parameters:
        df (pd.DataFrame): The dataframe containing data.
        exclude (list or None): Columns to exclude from plotting.
        filename (str): Path to save the figure.
        title (str): Figure title.
    """
    columns, n_cols_data = select_columns(df, exclude)

    # Prefer square layout
    n_rows = math.ceil(math.sqrt(n_cols_data))
    n_cols_grid = math.ceil(n_cols_data / n_rows)

    # If the square layout is too tall, use 4x3 aspect ratio instead
    if n_rows > n_cols_grid:
        n_cols_grid = min(4, n_cols_data)
        n_rows = math.ceil(n_cols_data / n_cols_grid)

    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(4*n_cols_grid, 4*n_rows))
    axes = axes.flatten()  # Flatten in case of single row/column

    for i, col in enumerate(columns):
        data = df[col].dropna()
        if len(data) == 0:
            print(f"Skipping column {col}: no valid data")
            continue

        stats.probplot(data, dist="norm", plot=axes[i])
        axes[i].set_title(col)
        axes[i].grid(True, alpha=0.3)

    # Turn off any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.savefig(filename)
    plt.close()
    print(f"Saved QQ-plot grid to {filename}")


def scatter_grid(df: pd.DataFrame,
                 exclude=None,
                 output_dir="data_analysis/figures/scatter",
                 title_prefix="Scatter Plots"):
    """
    Create scatter plot grids for each numeric column against all numeric columns, 
    including itself.

    Parameters:
        df (pd.DataFrame): DataFrame with numeric data.
        exclude (list or None): Columns to exclude from plotting.
        output_dir (str): Directory to save figures.
        title_prefix (str): Prefix for figure titles.
    """
    columns, n_cols_data = select_columns(df, exclude)
    
    # Loop over each column as X-axis
    for x_col in columns:
        y_cols = columns  # include self
        n_plots = len(y_cols)

        # Grid layout: square preferred
        n_rows = math.ceil(math.sqrt(n_plots))
        n_cols_grid = math.ceil(n_plots / n_rows)

        fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(4*n_cols_grid, 4*n_rows))
        axes = axes.flatten()

        for i, y_col in enumerate(y_cols):
            x_data = df[x_col].dropna()
            y_data = df[y_col].dropna()

            # Align indices
            mask = x_data.index.intersection(y_data.index)
            axes[i].scatter(x_data.loc[mask], y_data.loc[mask], alpha=0.6)
            axes[i].set_xlabel(x_col)
            axes[i].set_ylabel(y_col)
            axes[i].set_title(f"{y_col} vs {x_col}")
            axes[i].grid(True, alpha=0.3)

        # Turn off unused axes
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"{title_prefix}: {x_col} vs All", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_file = os.path.join(output_dir, f"{x_col}_scatter.png")
        plt.savefig(out_file)
        plt.close()
        print(f"Saved scatter plot grid for {x_col} to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Load evaluation CSV data and create QQ-plots.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the folder containing CSV files."
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated list of column names to exclude from QQ-plots."
    )

    args = parser.parse_args()

    exclude = [c.strip() for c in args.exclude.split(",")] if args.exclude else []

    print(f"Loading data from: {args.data}")
    data = load_csv_folder(args.data)

    print(f"Loaded {len(data)} CSV files:")
    
    base_out_path = "data_analysis/figures/"
    hist_out_path = f"{base_out_path}histograms/"
    qq_out_path = f"{base_out_path}qq_plots/"
    scatter_out_path = f"{base_out_path}scatter_plots/"
    path_list = [hist_out_path, qq_out_path, scatter_out_path]
    create_output_dirs(path_list)

    for filename, df in data.items():
        print(f" - {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        base_out_filename = filename.replace('.csv','')

        # Histogram
        hist_out_filename = f"{hist_out_path}/{base_out_filename}_histgrid.png"
        histogram_grid(df, exclude=exclude, filename=hist_out_filename,
                    title=f"Histograms: {filename}")

        # QQ-plot
        qq_out_filename = f"{qq_out_path}/{base_out_filename}_qqgrid.png"
        qqplot_grid(df, exclude=exclude, filename=qq_out_filename, 
                    title=f"QQ-Plots: {filename}")

        # Scatter plots: one figure per dependent variable
        scatter_file_dir = f"{scatter_out_path}/{base_out_filename}/"
        create_output_dirs([scatter_file_dir])

        scatter_grid(df, exclude=exclude, output_dir=scatter_file_dir,
                    title_prefix=f"Scatter Plots: {filename}")

if __name__ == "__main__":
    main()