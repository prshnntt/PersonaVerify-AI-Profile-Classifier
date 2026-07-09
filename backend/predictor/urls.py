"""
predictor/urls.py

URL routing for the predictor API.
"""

from django.urls import path
from .views import PredictView, HealthCheckView, StatsView, BulkPredictView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('health/', HealthCheckView.as_view(), name='health-check'),
    path('stats/', StatsView.as_view(), name='stats'),
    path('predict-bulk/', BulkPredictView.as_view(), name='predict-bulk'),
]
