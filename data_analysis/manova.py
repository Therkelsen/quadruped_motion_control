import argparse
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# ---------------------------
# Data loading
# ---------------------------

def load_csv_file(file_path):
    """Load a single CSV and return (filename, DataFrame)."""
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path)
    return filename, df


def load_csv_folder(folder_path: str):
    """Load all CSV files in folder using multithreading."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV files found.")

    data = {}
    file_paths = [os.path.join(folder_path, f) for f in csv_files]

    n_proc = os.cpu_count() or 1
    max_workers = max(1, n_proc - 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_csv_file, fp): fp for fp in file_paths}
        for future in as_completed(futures):
            filename, df = future.result()
            data[filename] = df

    return data


# ---------------------------
# MANOVA
# ---------------------------

def run_manova(data_dict, exclude=None):
    """Run MANOVA on all numeric variables across CSV groups."""
    all_columns = set()
    for df in data_dict.values():
        all_columns.update(df.select_dtypes(include="number").columns.tolist())

    if exclude:
        all_columns = [c for c in all_columns if c not in exclude]
    else:
        all_columns = list(all_columns)

    if not all_columns:
        print("No numeric columns found.")
        return

    # Combine CSVs
    combined_data = []
    for filename, df in data_dict.items():
        subset = df[all_columns].copy()
        subset["group"] = filename.replace(".csv", "")
        combined_data.append(subset)

    combined_df = pd.concat(combined_data, ignore_index=True)

    # Formula: DV1 + DV2 + ... ~ group
    formula = " + ".join(all_columns) + " ~ group"
    manova = MANOVA.from_formula(formula, data=combined_df)

    print("\n=== MANOVA RESULTS ===")
    print(manova.mv_test())


# ---------------------------
# ANOVA for each variable
# ---------------------------

def run_anovas(data_dict, exclude=None):
    """Run one-way ANOVA for each numeric dependent variable."""
    all_columns = set()
    for df in data_dict.values():
        all_columns.update(df.select_dtypes(include="number").columns.tolist())

    if exclude:
        all_columns = [c for c in all_columns if c not in exclude]
    else:
        all_columns = list(all_columns)

    combined_rows = []
    for filename, df in data_dict.items():
        tmp = df[all_columns].copy()
        tmp["group"] = filename.replace(".csv", "")
        combined_rows.append(tmp)

    combined_df = pd.concat(combined_rows, ignore_index=True)

    print("\n=== ANOVA RESULTS ===")

    for dv in all_columns:
        formula = f"{dv} ~ C(group)"
        model = smf.ols(formula, data=combined_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print(f"\n--- ANOVA for {dv} ---")
        print(anova_table)


# ---------------------------
# Tukey HSD
# ---------------------------

def run_tukey_hsd(data_dict, exclude=None):
    """Run Tukey HSD post-hoc analysis for each variable."""
    all_columns = set()
    for df in data_dict.values():
        all_columns.update(df.select_dtypes(include="number").columns.tolist())

    if exclude:
        all_columns = [c for c in all_columns if c not in exclude]
    else:
        all_columns = list(all_columns)

    combined_rows = []
    for filename, df in data_dict.items():
        tmp = df[all_columns].copy()
        tmp["group"] = filename.replace(".csv", "")
        combined_rows.append(tmp)

    combined_df = pd.concat(combined_rows, ignore_index=True)

    print("\n=== Tukey HSD RESULTS ===")

    for dv in all_columns:
        try:
            tukey = pairwise_tukeyhsd(
                endog=combined_df[dv],
                groups=combined_df["group"],
                alpha=0.05
            )
            print(f"\n--- Tukey HSD for {dv} ---")
            print(tukey)
        except Exception as e:
            print(f"Could not run Tukey for {dv}: {e}")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Folder with CSV files")
    parser.add_argument("--exclude", default="", help="Columns to exclude (comma-separated)")
    args = parser.parse_args()

    exclude = [x.strip() for x in args.exclude.split(",")] if args.exclude else []

    print(f"Loading data from {args.data}")
    data = load_csv_folder(args.data)

    print("\nRunning MANOVA...")
    run_manova(data, exclude)

    print("\nRunning ANOVAs...")
    run_anovas(data, exclude)

    print("\nRunning Tukey HSD...")
    run_tukey_hsd(data, exclude)


if __name__ == "__main__":
    main()
