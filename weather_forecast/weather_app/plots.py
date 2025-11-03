from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / 'static' / 'plots'


def _save(fig, name: str) -> str:
	PLOTS_DIR.mkdir(parents=True, exist_ok=True)
	file_path = PLOTS_DIR / name
	fig.tight_layout()
	fig.savefig(file_path, dpi=140)
	plt.close(fig)
	return f'/static/plots/{name}'


def plot_temperature_trend(df: pd.DataFrame) -> Optional[str]:
	if 'date' not in df.columns or 'temperature' not in df.columns:
		return None
	fig, ax = plt.subplots(figsize=(8, 3))
	ax.plot(df['date'], df['temperature'], color='#2563eb', linewidth=2)
	ax.set_title('Temperature Trend Over Time')
	ax.set_xlabel('Date')
	ax.set_ylabel('Temperature')
	ax.grid(alpha=0.3)
	return _save(fig, 'temperature_trend.png')


def plot_monthly_rainfall(monthly: pd.DataFrame) -> Optional[str]:
	if monthly.empty:
		return None
	fig, ax = plt.subplots(figsize=(8, 3))
	labels = monthly['year_month'].astype(str)
	ax.bar(labels, monthly['avg_rainfall'], color='#22c55e')
	ax.set_title('Average Monthly Rainfall')
	ax.set_xlabel('Month')
	ax.set_ylabel('Avg Rainfall')
	ax.tick_params(axis='x', rotation=45)
	return _save(fig, 'monthly_rainfall.png')


def plot_temp_vs_humidity(df: pd.DataFrame) -> Optional[str]:
	if 'temperature' not in df.columns or 'humidity' not in df.columns:
		return None
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.scatter(df['temperature'], df['humidity'], alpha=0.6, color='#f59e0b', edgecolors='white', linewidths=0.5)
	ax.set_title('Temperature vs Humidity')
	ax.set_xlabel('Temperature')
	ax.set_ylabel('Humidity')
	ax.grid(alpha=0.3)
	return _save(fig, 'temp_vs_humidity.png')


def plot_correlation_heatmap(corr: pd.DataFrame) -> Optional[str]:
	if corr.empty:
		return None
	fig, ax = plt.subplots(figsize=(5, 4))
	cax = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
	ax.set_xticks(np.arange(len(corr.columns)))
	ax.set_yticks(np.arange(len(corr.columns)))
	ax.set_xticklabels(corr.columns, rotation=45, ha='right')
	ax.set_yticklabels(corr.columns)
	fig.colorbar(cax)
	ax.set_title('Correlation Heatmap')
	return _save(fig, 'correlation_heatmap.png')


def plot_forecast(hist: pd.DataFrame, forecast: pd.DataFrame) -> Optional[str]:
	if hist.empty and forecast.empty:
		return None
	fig, ax = plt.subplots(figsize=(8, 3))
	if not hist.empty:
		ax.plot(hist['date'], hist['value'], label='Observed', color='#2563eb')
		if 'fit' in hist.columns:
			ax.plot(hist['date'], hist['fit'], label='Fit', color='#9333ea', linestyle='--')
	if not forecast.empty:
		ax.plot(forecast['date'], forecast['prediction'], label='Forecast', color='#ef4444')
	ax.legend()
	ax.set_title('Temperature Forecast (Linear Regression)')
	ax.set_xlabel('Date')
	ax.set_ylabel('Temperature')
	return _save(fig, 'forecast.png')


def plot_compare_temperature(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str) -> Optional[str]:
	if 'date' not in df1.columns or 'temperature' not in df1.columns:
		return None
	if 'date' not in df2.columns or 'temperature' not in df2.columns:
		return None
	fig, ax = plt.subplots(figsize=(8, 3))
	ax.plot(df1['date'], df1['temperature'], label=label1, color='#2563eb')
	ax.plot(df2['date'], df2['temperature'], label=label2, color='#ef4444')
	ax.legend()
	ax.set_title('Compare Temperature Trends')
	ax.set_xlabel('Date')
	ax.set_ylabel('Temperature')
	return _save(fig, 'compare_temp.png')


