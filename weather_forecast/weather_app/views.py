import os
import io
from datetime import datetime
from typing import Any, Dict

from django.http import JsonResponse, FileResponse, HttpRequest, HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

from .utils import (
	load_and_clean_data,
	compute_summary_stats,
	compute_correlations,
	detect_outliers_std,
	plot_temperature_trend,
	plot_monthly_rainfall,
	plot_temp_vs_humidity,
	plot_correlation_heatmap,
	simple_forecast_linear,
	DATA_PATH,
)


def _dataset_name() -> str:
	return os.path.basename(DATA_PATH)


def dashboard(request: HttpRequest) -> HttpResponse:
	try:
		df = load_and_clean_data()
	except Exception as e:
		return render(request, "dashboard.html", {"error": str(e), "dataset": _dataset_name()})

	stats = compute_summary_stats(df)
	plots = {
		"temp_trend": plot_temperature_trend(df),
		"monthly_rain": plot_monthly_rainfall(df),
	}
	context: Dict[str, Any] = {
		"dataset": _dataset_name(),
		"today": datetime.now(),
		"stats": stats,
		"plots": plots,
	}
	return render(request, "dashboard.html", context)


def analysis(request: HttpRequest) -> HttpResponse:
	try:
		df = load_and_clean_data()
	except Exception as e:
		return render(request, "analysis.html", {"error": str(e), "dataset": _dataset_name()})

	corr = compute_correlations(df)
	outliers_temp = detect_outliers_std(df, "temperature")
	plots = {
		"scatter": plot_temp_vs_humidity(df),
		"corr": plot_correlation_heatmap(df),
	}
	corr_headers = list(corr.columns) if not corr.empty else []
	corr_rows = []
	if not corr.empty:
		for idx, row in zip(corr.index.tolist(), corr.values.tolist()):
			corr_rows.append({"name": idx, "values": row})
	context = {
		"dataset": _dataset_name(),
		"today": datetime.now(),
		"corr_headers": corr_headers,
		"corr_rows": corr_rows,
		"outliers_count": len(outliers_temp),
		"plots": plots,
	}
	return render(request, "analysis.html", context)


def forecast(request: HttpRequest) -> HttpResponse:
	try:
		df = load_and_clean_data()
	except Exception as e:
		return render(request, "forecast.html", {"error": str(e), "dataset": _dataset_name()})

	pred_df = simple_forecast_linear(df, days=7)
	context = {
		"dataset": _dataset_name(),
		"today": datetime.now(),
		"predictions": pred_df.to_dict(orient="records"),
	}
	return render(request, "forecast.html", context)


@csrf_exempt
def compare(request: HttpRequest) -> HttpResponse:
	context: Dict[str, Any] = {"dataset": _dataset_name(), "today": datetime.now()}
	if request.method == "POST":
		files = [request.FILES.get("file1"), request.FILES.get("file2")]
		labels = ["Dataset A", "Dataset B"]
		results = []
		for f, label in zip(files, labels):
			if not f:
				results.append({"label": label, "error": "No file uploaded"})
				continue
			try:
				df = load_and_clean_data(path=f)
				stats = compute_summary_stats(df)
				trend = plot_temperature_trend(df)
				results.append({"label": label, "stats": stats, "trend": trend})
			except Exception as e:
				results.append({"label": label, "error": str(e)})
		context["results"] = results
	return render(request, "compare.html", context)


def api_summary(request: HttpRequest) -> JsonResponse:
	try:
		df = load_and_clean_data()
		stats = compute_summary_stats(df)
		corr = compute_correlations(df)
		return JsonResponse({
			"dataset": _dataset_name(),
			"stats": stats,
			"correlations": corr.to_dict() if not corr.empty else {},
		})
	except Exception as e:
		return JsonResponse({"error": str(e)}, status=400)


def download_report_pdf(request: HttpRequest) -> FileResponse:
	try:
		df = load_and_clean_data()
		stats = compute_summary_stats(df)
		plot_paths = [
			plot_temperature_trend(df),
			plot_monthly_rainfall(df),
			plot_temp_vs_humidity(df),
			plot_correlation_heatmap(df),
		]
		buffer = io.BytesIO()
		c = canvas.Canvas(buffer, pagesize=A4)
		width, height = A4
		c.setTitle("Weather Report")
		c.setFont("Helvetica-Bold", 16)
		c.drawString(72, height - 72, "Weather Report")
		c.setFont("Helvetica", 10)
		c.drawString(72, height - 90, f"Dataset: {_dataset_name()}  |  Generated: {datetime.now():%Y-%m-%d %H:%M}")
		c.line(72, height - 96, width - 72, height - 96)

		# Stats section
		y = height - 120
		c.setFont("Helvetica-Bold", 12)
		c.drawString(72, y, "Key Statistics")
		y -= 14
		c.setFont("Helvetica", 10)
		for key, vals in stats.items():
			c.drawString(72, y, f"{key.capitalize()}: mean={vals['mean']:.2f}, min={vals['min']:.2f}, max={vals['max']:.2f}, std={vals['std']:.2f}")
			y -= 12

		# Charts
		y -= 8
		c.setFont("Helvetica-Bold", 12)
		c.drawString(72, y, "Charts")
		y -= 12
		for rel_path in plot_paths:
			if not rel_path:
				continue
			abs_path = os.path.join(os.path.dirname(__file__), "static", rel_path.replace("/", os.sep))
			try:
				c.drawImage(abs_path, 72, y - 3.0*inch, width=4.5*inch, height=3.0*inch, preserveAspectRatio=True, anchor='sw')
				y -= 3.2*inch
				if y < 144:
					c.showPage()
					y = height - 72
			except Exception:
				continue

		c.showPage()
		c.save()
		buffer.seek(0)
		filename = "weather_report.pdf"
		return FileResponse(buffer, as_attachment=True, filename=filename)
	except Exception as e:
		# Fallback simple PDF with error
		buffer = io.BytesIO()
		c = canvas.Canvas(buffer, pagesize=A4)
		c.setFont("Helvetica", 12)
		c.drawString(72, 800, f"Failed to generate report: {str(e)}")
		c.showPage()
		c.save()
		buffer.seek(0)
		return FileResponse(buffer, as_attachment=True, filename="weather_report_error.pdf")


