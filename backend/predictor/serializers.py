"""
predictor/serializers.py

Defines the request/response schemas for the prediction API.
"""

from rest_framework import serializers


class ProfileInputSerializer(serializers.Serializer):
    """
    Validates the incoming profile features.
    """
    profile_pic = serializers.IntegerField(
        min_value=0, max_value=1,
        help_text="1 if user has a profile picture, else 0"
    )
    nums_length_username = serializers.FloatField(
        min_value=0.0, max_value=1.0,
        help_text="Ratio of numeric chars to total username length"
    )
    fullname_words = serializers.IntegerField(
        min_value=0,
        help_text="Number of words in the full name"
    )
    nums_length_fullname = serializers.FloatField(
        min_value=0.0, max_value=1.0,
        help_text="Ratio of numeric chars in full name"
    )
    name_eq_username = serializers.IntegerField(
        min_value=0, max_value=1,
        help_text="1 if display name equals username, else 0"
    )
    description_length = serializers.IntegerField(
        min_value=0,
        help_text="Character count of bio/description"
    )
    external_url = serializers.IntegerField(
        min_value=0, max_value=1,
        help_text="1 if profile has an external URL, else 0"
    )
    private = serializers.IntegerField(
        min_value=0, max_value=1,
        help_text="1 if account is private, else 0"
    )
    posts = serializers.IntegerField(
        min_value=0,
        help_text="Total number of posts"
    )
    followers = serializers.IntegerField(
        min_value=0,
        help_text="Number of followers"
    )
    follows = serializers.IntegerField(
        min_value=0,
        help_text="Number of accounts the user follows"
    )


# ─── Explainability Sub-serializers ──────────────────────────────────────────

class FeatureContributionSerializer(serializers.Serializer):
    """A single feature's contribution to the prediction."""
    feature = serializers.CharField(help_text="Internal feature name")
    label = serializers.CharField(help_text="Human-readable feature name")
    contribution = serializers.FloatField(help_text="How much this feature shifted the prediction (+/- float)")
    direction = serializers.CharField(help_text="'toward_fake', 'toward_real', or 'neutral'")
    value = serializers.FloatField(help_text="Actual value of this feature for the input profile")


class RiskFactorSerializer(serializers.Serializer):
    """A human-readable risk indicator."""
    direction = serializers.CharField(help_text="'toward_fake' or 'toward_real'")
    message = serializers.CharField(help_text="Plain-English explanation")


class GlobalImportanceSerializer(serializers.Serializer):
    """Global feature importance from the trained model."""
    feature = serializers.CharField()
    label = serializers.CharField()
    importance = serializers.FloatField()


class ExplainabilitySerializer(serializers.Serializer):
    """Full Explainable AI output."""
    method = serializers.CharField(help_text="XAI method used")
    model = serializers.CharField(help_text="Model type")
    bias = serializers.FloatField(help_text="Base rate prediction before any features are considered")
    feature_contributions = FeatureContributionSerializer(many=True)
    risk_factors = RiskFactorSerializer(many=True)
    global_feature_importance = GlobalImportanceSerializer(many=True)


class PredictionDetailsSerializer(serializers.Serializer):
    raw_prediction = serializers.IntegerField()
    probability_real = serializers.FloatField()
    probability_fake = serializers.FloatField()


# ─── Main Output Serializer ──────────────────────────────────────────────────

class PredictionOutputSerializer(serializers.Serializer):
    """
    Full prediction response with Explainable AI.
    """
    prediction = serializers.CharField(help_text="'Fake' or 'Real'")
    confidence_score = serializers.FloatField(help_text="Probability (0-1)")
    top_features = serializers.ListField(
        child=serializers.CharField(),
        help_text="Top 5 contributing features (human-readable)"
    )
    details = PredictionDetailsSerializer()
    explainability = ExplainabilitySerializer(
        help_text="Full Explainable AI breakdown"
    )
