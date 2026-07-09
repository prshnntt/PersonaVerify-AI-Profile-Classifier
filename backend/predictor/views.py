"""
predictor/views.py

API views for the PersonaVerify prediction service.
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from .serializers import ProfileInputSerializer, PredictionOutputSerializer
from .services import predict_profile


class PredictView(APIView):
    """
    POST /api/predict/

    Accepts profile features as JSON and returns a prediction
    indicating whether the profile is Fake or Real, along with
    a confidence score and top contributing features.

    ---
    Request body:
    {
        "profile_pic": 1,
        "nums_length_username": 0.27,
        "fullname_words": 2,
        "nums_length_fullname": 0.0,
        "name_eq_username": 0,
        "description_length": 53,
        "external_url": 0,
        "private": 0,
        "posts": 32,
        "followers": 1000,
        "follows": 955
    }

    Response:
    {
        "prediction": "Fake" or "Real",
        "confidence_score": 0.82,
        "top_features": ["low_followers", "high_following", ...],
        "details": {
            "raw_prediction": 1,
            "probability_real": 0.18,
            "probability_fake": 0.82
        }
    }
    """

    def post(self, request):
        # Step 1: Validate input
        serializer = ProfileInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {'error': 'Invalid input', 'details': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Step 2: Run prediction
        try:
            result = predict_profile(serializer.validated_data)
        except Exception as e:
            return Response(
                {'error': 'Prediction failed', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Step 3: Serialize and return
        output = PredictionOutputSerializer(result)
        return Response(output.data, status=status.HTTP_200_OK)


class HealthCheckView(APIView):
    """
    GET /api/health/

    Returns the health status of the API and whether the ML model is loaded.
    """

    def get(self, request):
        from .model_loader import ModelLoader

        return Response({
            'status': 'ok',
            'model_loaded': ModelLoader._is_loaded,
            'model_type': type(ModelLoader.get_model()).__name__
                if ModelLoader._is_loaded else None,
        })


class StatsView(APIView):
    """
    GET /api/stats/
    Returns the total prediction counts from the database.
    """
    def get(self, request):
        from .models import PredictionLog
        
        total = PredictionLog.objects.count()
        fake_count = PredictionLog.objects.filter(prediction='Fake').count()
        real_count = PredictionLog.objects.filter(prediction='Real').count()
        
        return Response({
            'total_predictions': total,
            'fake_predictions': fake_count,
            'real_predictions': real_count
        })


class BulkPredictView(APIView):
    """
    POST /api/predict-bulk/
    Accepts a multipart file upload containing a CSV.
    """
    parser_classes = [MultiPartParser]

    def post(self, request):
        from .services import predict_bulk_csv
        
        if 'file' not in request.FILES:
            return Response({'error': 'No file uploaded. Key must be "file".'}, status=status.HTTP_400_BAD_REQUEST)
            
        file = request.FILES['file']
        
        if not file.name.endswith('.csv'):
            return Response({'error': 'File must be a CSV.'}, status=status.HTTP_400_BAD_REQUEST)
            
        try:
            results = predict_bulk_csv(file)
            return Response(results, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': f"Internal server error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

