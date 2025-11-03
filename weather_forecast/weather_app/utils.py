import io
import os
import time
from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "weather_app", "static")
# Prefer user's dataset if present
USER_DATASET_PATH = r"c:\Users\ASUS\Downloads\weather_forecast_data.csv"
DEFAULT_DATA_PATH = os.path.join(STATIC_DIR, "data", "weather_data.csv")
DATA_PATH = USER_DATASET_PATH if os.path.exists(USER_DATASET_PATH) else DEFAULT_DATA_PATH
PLOTS_DIR = os.path.join(STATIC_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
	mapping = {
		"date": ["date", "Date", "day"],
		"temperature": ["temperature", "temp", "temp_c", "mean_temp", "avg_temp"],
		"humidity": ["humidity", "hum", "relative_humidity"],
		"pressure": ["pressure", "press", "barometric_pressure"],
		"rainfall": ["rainfall", "rain", "precipitation", "precip"],
		"wind_speed": ["wind_speed", "wind", "windspd", "wind_speed_kmh"],
	}
	df_cols = {c.lower(): c for c in df.columns}
	normalized = {}
	for target, aliases in mapping.items():
		for alias in aliases:
			if alias.lower() in df_cols:
				normalized[target] = df_cols[alias.lower()]
				break
	
	renamed = {
		v: k
		for k, v in normalized.items()
	}
	return df.rename(columns=renamed)


def load_and_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
	df = pd.read_csv(path)
	df = _normalize_columns(df)
	if "date" not in df.columns:
		raise ValueError("Input dataset must include a date column (detected after normalization).")
	
	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df = df.dropna(subset=["date"])  # must have date
	
	for col in ["temperature", "humidity", "pressure", "rainfall", "wind_speed"]:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	
	# Strategy: fill numeric NaNs with column median, then drop remaining empty rows
	numeric_cols = [c for c in ["temperature", "humidity", "pressure", "rainfall", "wind_speed"] if c in df.columns]
	for c in numeric_cols:
		median_val = df[c].median()
		df[c] = df[c].fillna(median_val)
	
	df = df.dropna()
	df = df.sort_values("date").reset_index(drop=True)
	return df


def compute_summary_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
	stats: Dict[str, Dict[str, float]] = {}
	for col in ["temperature", "humidity", "rainfall"]:
		if col in df.columns:
			series = df[col].dropna()
			stats[col] = {
				"mean": float(series.mean()),
				"max": float(series.max()),
				"min": float(series.min()),
				"std": float(series.std(ddof=0)) if len(series) > 0 else 0.0,
			}
	return stats


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
	cols = [c for c in ["temperature", "humidity", "pressure", "rainfall", "wind_speed"] if c in df.columns]
	if not cols:
		return pd.DataFrame()
	return df[cols].corr()


def detect_outliers_std(df: pd.DataFrame, column: str, z_threshold: float = 3.0) -> pd.DataFrame:
	if column not in df.columns:
		return pd.DataFrame(columns=df.columns)
	series = df[column]
	mu = series.mean()
	sigma = series.std(ddof=0)
	if sigma == 0 or np.isnan(sigma):
		return pd.DataFrame(columns=df.columns)
	z = (series - mu) / sigma
	mask = np.abs(z) > z_threshold
	return df.loc[mask]


def _timestamped_filename(prefix: str) -> str:
	return f"{prefix}_{int(time.time()*1000)}.png"


def plot_temperature_trend(df: pd.DataFrame) -> str:
	if "temperature" not in df.columns:
		return ""
	plt.figure(figsize=(10, 4))
	plt.plot(df["date"], df["temperature"], color="#1d4ed8")
	plt.title("Temperature Trend Over Time")
	plt.xlabel("Date")
	plt.ylabel("Temperature")
	plt.grid(alpha=0.3)
	plt.tight_layout()
	fname = _timestamped_filename("temp_trend")
	fpath = os.path.join(PLOTS_DIR, fname)
	plt.savefig(fpath, dpi=110)
	plt.close()
	return f"plots/{fname}"


def plot_monthly_rainfall(df: pd.DataFrame) -> str:
	if "rainfall" not in df.columns:
		return ""
	df2 = df.copy()
	df2["month"] = df2["date"].dt.to_period("M").dt.to_timestamp()
	monthly = df2.groupby("month")["rainfall"].mean().reset_index()
	plt.figure(figsize=(10, 4))
	plt.bar(monthly["month"], monthly["rainfall"], color="#10b981")
	plt.title("Average Monthly Rainfall")
	plt.xlabel("Month")
	plt.ylabel("Rainfall")
	plt.xticks(rotation=45)
	plt.grid(axis="y", alpha=0.3)
	plt.tight_layout()
	fname = _timestamped_filename("monthly_rain")
	fpath = os.path.join(PLOTS_DIR, fname)
	plt.savefig(fpath, dpi=110)
	plt.close()
	return f"plots/{fname}"


def plot_temp_vs_humidity(df: pd.DataFrame) -> str:
	if not {"temperature", "humidity"}.issubset(df.columns):
		return ""
	plt.figure(figsize=(6, 5))
	plt.scatter(df["temperature"], df["humidity"], alpha=0.6, color="#f59e0b")
	plt.title("Temperature vs Humidity")
	plt.xlabel("Temperature")
	plt.ylabel("Humidity")
	plt.grid(alpha=0.3)
	plt.tight_layout()
	fname = _timestamped_filename("temp_hum_scatter")
	fpath = os.path.join(PLOTS_DIR, fname)
	plt.savefig(fpath, dpi=110)
	plt.close()
	return f"plots/{fname}"


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
	corr = compute_correlations(df)
	if corr.empty:
		return ""
	plt.figure(figsize=(6, 5))
	sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
	plt.title("Correlation Heatmap")
	plt.tight_layout()
	fname = _timestamped_filename("corr_heatmap")
	fpath = os.path.join(PLOTS_DIR, fname)
	plt.savefig(fpath, dpi=110)
	plt.close()
	return f"plots/{fname}"


def simple_forecast_linear(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
	if "temperature" not in df.columns:
		return pd.DataFrame(columns=["date", "predicted_temperature"]) 
	df = df.dropna(subset=["temperature"]).copy()
	df["t"] = (df["date"] - df["date"].min()).dt.days
	x = df["t"].to_numpy(dtype=float)
	y = df["temperature"].to_numpy(dtype=float)
	if len(x) < 2:
		return pd.DataFrame(columns=["date", "predicted_temperature"]) 
	coeffs = np.polyfit(x, y, deg=1)
	m, b = coeffs[0], coeffs[1]
	last_day = int(df["t"].max())
	future_days = np.arange(last_day + 1, last_day + 1 + days)
	preds = m * future_days + b
	start_date = df["date"].min()
	future_dates = [start_date + timedelta(days=int(d)) for d in future_days]
	return pd.DataFrame({"date": future_dates, "predicted_temperature": preds})


