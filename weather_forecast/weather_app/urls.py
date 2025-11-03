from django.urls import path
from . import views

urlpatterns = [
	path("", views.dashboard, name="dashboard"),
	path("analysis/", views.analysis, name="analysis"),
	path("forecast/", views.forecast, name="forecast"),
	path("compare/", views.compare, name="compare"),
	path("api/summary/", views.api_summary, name="api_summary"),
	path("report/pdf/", views.download_report_pdf, name="download_report_pdf"),
]


