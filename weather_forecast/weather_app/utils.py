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
		"date": ["date", "Date", "day", "timestamp", "datetime"],
		"temperature": ["temperature", "temp", "temp_c", "mean_temp", "avg_temp"],
		"humidity": ["humidity", "hum", "relative_humidity"],
		"pressure": ["pressure", "press", "barometric_pressure"],
		"rainfall": ["rainfall", "rain", "precipitation", "precip"],
		"wind_speed": ["wind_speed", "wind", "windspd", "wind_speed_kmh", "wind speed"],
	}
	df_cols = {c.lower(): c for c in df.columns}
	normalized = {}
	for target, aliases in mapping.items():
		for alias in aliases:
			if alias.lower() in df_cols:
				normalized[target] = df_cols[alias.lower()]
				break
	
	renamed = {v: k for k, v in normalized.items()}
	return df.rename(columns=renamed)


def load_and_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
	# Accept both file paths and Django UploadedFile objects
	if hasattr(path, "read"):
		# file-like
		df = pd.read_csv(path)
	else:
		df = pd.read_csv(path)

	df = _normalize_columns(df)

	# Create a date column if missing by using a synthetic daily index
	if "date" not in df.columns:
		start = pd.Timestamp("2006-01-01")
		idx = pd.date_range(start=start, periods=len(df), freq="D")
		df.insert(0, "date", idx)
	else:
		df["date"] = pd.to_datetime(df["date"], errors="coerce")
		df = df.dropna(subset=["date"])  # must have date if present

	# Coerce numeric columns
	for col in ["temperature", "humidity", "pressure", "rainfall", "wind_speed"]:
		if col in df.columns:
			# Special handling for rainfall that may be categorical (e.g., 'rain'/'no rain')
			if col == "rainfall" and df[col].dtype == object:
				lower = df[col].astype(str).str.strip().str.lower()
				df[col] = np.where(lower.isin(["rain", "raining", "yes", "1", "true"]), 1.0,
								 np.where(lower.isin(["no rain", "no", "0", "false", "clear"]), 0.0, np.nan))
			else:
				df[col] = pd.to_numeric(df[col], errors="coerce")

	# Fill numeric NaNs with column median, then drop remaining empties
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
	# Downsample for readability on large datasets
	df_ts = df[["date", "temperature"]].copy()
	if len(df_ts) > 400:
		weekly = df_ts.set_index("date").resample("W").mean().reset_index()
		plot_df = weekly
		rolling = plot_df["temperature"].rolling(window=6, min_periods=1).mean()
	else:
		plot_df = df_ts
		rolling = plot_df["temperature"].rolling(window=7, min_periods=1).mean()
	plt.figure(figsize=(10, 4))
	plt.plot(plot_df["date"], plot_df["temperature"], color="#93c5fd", linewidth=1)
	plt.plot(plot_df["date"], rolling, color="#1d4ed8", linewidth=2)
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
	# Use recent window to capture latest trend and avoid flattening
	window_days = 90
	df_recent = df.dropna(subset=["temperature"]).copy()
	if len(df_recent) > window_days:
		df_recent = df_recent.iloc[-window_days:]
	# Scale time to [-1, 1] for numerical stability
	t = (df_recent["date"] - df_recent["date"].min()).dt.days.astype(float)
	if t.max() == 0:
		# Not enough variation, return constant mean
		mean_val = float(df_recent["temperature"].mean())
		future_dates = [df_recent["date"].max() + timedelta(days=i) for i in range(1, days + 1)]
		return pd.DataFrame({"date": future_dates, "predicted_temperature": [mean_val] * days})
	t_scaled = 2 * (t - t.min()) / (t.max() - t.min()) - 1
	y = df_recent["temperature"].to_numpy(dtype=float)
	# Fit quadratic; fallback to linear if singular
	try:
		coeffs = np.polyfit(t_scaled, y, deg=2)
		poly = np.poly1d(coeffs)
	except Exception:
		coeffs = np.polyfit(t_scaled, y, deg=1)
		poly = np.poly1d(coeffs)
	last_t = t.max()
	future_raw = np.arange(last_t + 1, last_t + 1 + days, dtype=float)
	future_scaled = 2 * (future_raw - t.min()) / (t.max() - t.min()) - 1
	preds = poly(future_scaled)
	# Clamp predictions to a sensible range based on recent history
	y_min, y_max = float(np.min(y)), float(np.max(y))
	pad = max(1.0, 0.15 * (y_max - y_min))
	preds = np.clip(preds, y_min - pad, y_max + pad)
	start_date = df_recent["date"].min()
	future_dates = [start_date + timedelta(days=int(d)) for d in future_raw]
	return pd.DataFrame({"date": future_dates, "predicted_temperature": preds})


