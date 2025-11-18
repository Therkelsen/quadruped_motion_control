"""Create publication-quality comparison plots from TensorBoard CSV exports.

This script reads CSV exports from `data_analysis/tensorboard/reward/` and
`data_analysis/tensorboard/length/`. Each CSV is expected to have the header
"Wall time,Step,Value" (the standard TensorBoard CSV export format).

It produces two main figures (saved as PNG and PDF in `data_analysis/plots/`):
 - mean_rewards_normalized.(png|pdf): per-model mean reward curves, min-max
   normalized per model to allow direct visual comparison between models with
   different reward scales; mean +/- std shaded.
 - episode_lengths.(png|pdf): per-model mean episode length curves (raw
   values) with mean +/- std shaded.

Usage: run the script from the repository root::
	python data_analysis/tensorboard_plots.py

The script will automatically create `data_analysis/plots/` if it doesn't
exist.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def find_csvs(metric_dir: str) -> List[str]:
	"""Return list of CSV files under metric_dir (non-recursive)."""
	pattern = os.path.join(metric_dir, "*.csv")
	return sorted(glob.glob(pattern))


def model_name_from_filename(path: str) -> str:
	"""Infer a model name from filename (e.g., 'SAC.csv' -> 'SAC')."""
	base = os.path.basename(path)
	name, _ = os.path.splitext(base)
	return name


def load_metric_csv(path: str) -> pd.DataFrame:
	"""Load a TensorBoard CSV and return DataFrame with columns ['Step','Value'].

	Ensures Step is integer and Value is float.
	"""
	df = pd.read_csv(path)
	# Expected columns: Wall time,Step,Value
	if 'Step' not in df.columns or 'Value' not in df.columns:
		raise ValueError(f"CSV {path} missing required columns: {df.columns.tolist()}")
	df = df[['Step', 'Value']].dropna()
	df = df.sort_values('Step')
	df['Step'] = df['Step'].astype(int)
	df['Value'] = df['Value'].astype(float)
	return df


def aggregate_runs_to_grid(runs: List[pd.DataFrame], grid: np.ndarray) -> np.ndarray:
	"""Interpolate each run to the common grid and return array shape (n_runs, len(grid)).

	Missing values beyond a run's max Step are filled with NaN.
	"""
	arr = np.full((len(runs), len(grid)), np.nan, dtype=float)
	for i, df in enumerate(runs):
		steps = df['Step'].values
		values = df['Value'].values
		if len(steps) == 0:
			continue
		# For interpolation we only consider the range [min_step, max_step]
		min_s, max_s = steps[0], steps[-1]
		mask = (grid >= min_s) & (grid <= max_s)
		if mask.any():
			arr[i, mask] = np.interp(grid[mask], steps, values)
	return arr


def compute_mean_std_from_runs(runs: List[pd.DataFrame], n_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Given list of runs (DataFrames), compute a common step grid and mean/std arrays.

	Returns (grid, mean, std).
	"""
	# Build global grid across runs: choose 200 evenly spaced points between min and max Step
	all_steps = np.concatenate([r['Step'].values for r in runs if len(r) > 0])
	if len(all_steps) == 0:
		raise ValueError("No step data found in runs")
	min_step, max_step = int(all_steps.min()), int(all_steps.max())
	if min_step == max_step:
		grid = np.array([min_step])
	else:
		grid = np.linspace(min_step, max_step, n_points, dtype=int)

	arr = aggregate_runs_to_grid(runs, grid)
	mean = np.nanmean(arr, axis=0)
	std = np.nanstd(arr, axis=0)
	return grid, mean, std


def ensure_plot_dir(out_dir: str) -> None:
	os.makedirs(out_dir, exist_ok=True)


def plot_metric(models_runs: Dict[str, List[pd.DataFrame]], out_path_base: str, *,
				title: str, xlabel: str = 'Step', ylabel: str = '', normalize: bool = False):
	"""Plot mean +/- std for each model in models_runs.

	models_runs: dict mapping model_name -> list of DataFrames (each with Step,Value)
	out_path_base: file path without extension (will save .png and .pdf)
	normalize: if True, min-max normalize each model's mean curve to [0,1].
	"""
	sns.set(style='whitegrid', context='paper', rc={'font.size': 12, 'axes.titlesize': 14})
	plt.figure(figsize=(7, 4.5))

	color_cycle = sns.color_palette('tab10')
	for idx, (model, runs) in enumerate(models_runs.items()):
		if len(runs) == 0:
			continue
		try:
			grid, mean, std = compute_mean_std_from_runs(runs)
		except ValueError:
			continue

		if normalize:
			# min-max normalize mean curve for plotting comparability
			mm_min, mm_max = np.nanmin(mean), np.nanmax(mean)
			if np.isfinite(mm_min) and np.isfinite(mm_max) and mm_max > mm_min:
				mean_plot = (mean - mm_min) / (mm_max - mm_min)
				std_plot = std / (mm_max - mm_min)
				ylabel_plot = ylabel + ' (min-max normalized)'
			else:
				mean_plot, std_plot = mean, std
				ylabel_plot = ylabel
		else:
			mean_plot, std_plot = mean, std
			ylabel_plot = ylabel

		color = color_cycle[idx % len(color_cycle)]
		plt.plot(grid, mean_plot, label=model, color=color, linewidth=1.8)
		plt.fill_between(grid, mean_plot - std_plot, mean_plot + std_plot, color=color, alpha=0.22)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel_plot if normalize else ylabel)
	plt.title(title)
	plt.legend(title='Model')
	plt.tight_layout()

	png_path = out_path_base + '.png'
	pdf_path = out_path_base + '.pdf'
	plt.savefig(png_path, dpi=300)
	plt.savefig(pdf_path)
	plt.close()


def build_models_runs(metric_folder: str) -> Dict[str, List[pd.DataFrame]]:
	"""Scan a folder of CSVs and return mapping model->list of runs (DataFrames)."""
	files = find_csvs(metric_folder)
	models = defaultdict(list)
	for p in files:
		model = model_name_from_filename(p)
		try:
			df = load_metric_csv(p)
			models[model].append(df)
		except Exception as e:
			print(f"Warning: failed to load {p}: {e}")
	return models


def main(out_dir: str = 'data_analysis/plots') -> None:
	ensure_plot_dir(out_dir)

	# Reward: produce a min-max normalized mean reward plot for comparability
	reward_folder = os.path.join('data_analysis', 'tensorboard', 'reward')
	reward_models = build_models_runs(reward_folder)
	if reward_models:
		out_base = os.path.join(out_dir, 'mean_rewards_normalized')
		plot_metric(reward_models, out_base, title='Mean Reward (min-max normalized per model)',
					xlabel='Step', ylabel='Reward', normalize=True)
		# Also save a raw mean rewards plot (auxiliary)
		out_base_raw = os.path.join(out_dir, 'mean_rewards_raw')
		plot_metric(reward_models, out_base_raw, title='Mean Reward',
					xlabel='Step', ylabel='Reward', normalize=False)
		print(f"Saved reward plots to {out_dir}")
	else:
		print(f"No reward CSVs found in {reward_folder}")

	# Episode length: raw mean plot
	length_folder = os.path.join('data_analysis', 'tensorboard', 'length')
	length_models = build_models_runs(length_folder)
	if length_models:
		out_base = os.path.join(out_dir, 'episode_lengths')
		plot_metric(length_models, out_base, title='Episode Length', xlabel='Step', ylabel='Episode length', normalize=False)
		print(f"Saved episode length plot to {out_dir}")
	else:
		print(f"No length CSVs found in {length_folder}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create plots from TensorBoard CSV exports')
	parser.add_argument('--out-dir', default='data_analysis/plots', help='Directory to save plots')
	args = parser.parse_args()
	main(args.out_dir)

