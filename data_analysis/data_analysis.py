import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
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


def qqplot_total_reward(df, filename="data_analysis/figures/total_reward_qq.png", title="QQ-Plot: total_reward"):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    data = df["total_reward"].dropna()

    plt.figure(figsize=(6, 6))
    stats.probplot(data, dist="norm", plot=plt)

    plt.title(title)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()  # Close the figure to free memory
    print(f"Saved QQ-plot to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Load evaluation CSV data.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the folder containing CSV files."
    )

    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    data = load_csv_folder(args.data)

    print(f"Loaded {len(data)} CSV files:")
    for filename, df in data.items():
        print(f" - {filename}: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nExample preview of first file:")
    first_key = list(data.keys())[0]
    print(data[first_key].head())

    qqplot_total_reward(df)


if __name__ == "__main__":
    main()
