# Weather Forecast Data Dashboard (Django + Pandas + NumPy + Matplotlib)

A full-stack weather analytics dashboard using Django, Pandas, NumPy, and Matplotlib. It loads a Kaggle CSV from `weather_app/static/data/`, cleans and analyzes the data, renders charts to `weather_app/static/plots/`, and displays a modern responsive UI.

## Features
- Load historical weather CSV (date, temperature, humidity, pressure, rainfall, wind_speed)
- Cleaning: fill/drop missing values
- Stats: mean, max, min, std; correlations; outlier detection
- Visuals: temperature trend (line), monthly rainfall (bar), temp vs humidity (scatter), correlation heatmap
- 7-day simple forecast (NumPy linear regression)
- Download PDF report (ReportLab)
- JSON API endpoint for summarized stats
- Optional compare view to upload and compare two datasets

## Quickstart

1) Create and activate a virtual environment
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Run the server
```bash
python weather_forecast/manage.py runserver
```

4) Visit
- Dashboard: http://127.0.0.1:8000/
- Analysis: http://127.0.0.1:8000/analysis/
- Forecast: http://127.0.0.1:8000/forecast/
- Compare: http://127.0.0.1:8000/compare/
- API: http://127.0.0.1:8000/api/summary/

## Dataset
Place a Kaggle CSV as `weather_app/static/data/weather_data.csv`.
Example compatible datasets:
- `Daily Weather Dataset`
- `Weather Data from 2006–2016`

Required columns (case-insensitive supported): `date`, `temperature`, `humidity`, `pressure`, `rainfall`, `wind_speed`.

A tiny sample is included at `weather_app/static/data/weather_data.csv` for demo only. Replace with a real Kaggle file for meaningful insights.

## Notes
- Generated plots are saved to `weather_app/static/plots/` with timestamped filenames to avoid caching issues.
- Bootstrap 5 CDN is used for responsive UI.
- If PDF fonts render oddly, install system fonts or adjust ReportLab font usage.

## Screenshots
Run locally and take screenshots of `dashboard`, `analysis`, and `forecast` pages.

