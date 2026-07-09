"""
predictor/model_loader.py

Singleton-pattern model loader.
Loads the trained ML models, scaler, feature names, and feature importances
ONCE at server startup and caches them in memory for fast inference.
"""

import joblib
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and caches all ML artifacts needed for prediction and explainability.
    Uses class-level attributes so the model is loaded only once
    across all requests (singleton pattern).
    """
    _model = None           # Primary model (Random Forest — for predictions + XAI)
    _scaler = None
    _feature_names = None
    _feature_importances = None
    _is_loaded = False

    @classmethod
    def load(cls):
        """Load all ML artifacts from disk into memory."""
        if cls._is_loaded:
            return

        model_dir = settings.ML_MODEL_DIR

        try:
            # Use Random Forest as the primary model
            # — same accuracy as Logistic Regression but higher AUC-ROC (99.36%)
            # — also provides tree-based feature importances for Explainable AI
            cls._model = joblib.load(model_dir / 'random_forest.pkl')
            cls._scaler = joblib.load(model_dir / 'scaler.pkl')
            cls._feature_names = joblib.load(model_dir / 'feature_names.pkl')
            cls._feature_importances = joblib.load(model_dir / 'feature_importances.pkl')
            cls._is_loaded = True
            logger.info(
                f"ML model loaded: {type(cls._model).__name__} "
                f"with {len(cls._feature_names)} features."
            )
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            raise RuntimeError(f"Could not load ML model artifacts: {e}")

    @classmethod
    def get_model(cls):
        if not cls._is_loaded:
            cls.load()
        return cls._model

    @classmethod
    def get_scaler(cls):
        if not cls._is_loaded:
            cls.load()
        return cls._scaler

    @classmethod
    def get_feature_names(cls):
        if not cls._is_loaded:
            cls.load()
        return cls._feature_names

    @classmethod
    def get_feature_importances(cls):
        if not cls._is_loaded:
            cls.load()
        return cls._feature_importances
