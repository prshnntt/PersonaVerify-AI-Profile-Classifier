"""
predictor/apps.py

App configuration. Pre-loads the ML model at Django startup.
"""

from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    def ready(self):
        """Load ML model into memory when the server starts."""
        from .model_loader import ModelLoader
        ModelLoader.load()
