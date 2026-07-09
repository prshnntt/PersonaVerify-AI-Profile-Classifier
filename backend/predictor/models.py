from django.db import models

class PredictionLog(models.Model):
    """Stores the result of every prediction for dashboard analytics."""
    prediction = models.CharField(max_length=10) # 'Fake' or 'Real'
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prediction} - {self.confidence:.2f} ({self.created_at})"
