"""
predictor/services.py

Business logic layer: feature engineering, prediction, and explainability.
Keeps the view thin and the logic testable.
"""

import numpy as np
import pandas as pd
from .model_loader import ModelLoader
from .explainer import explain_prediction
from .models import PredictionLog


# ─── Feature engineering ─────────────────────────────────────────────────────

def engineer_features(raw: dict) -> pd.DataFrame:
    """
    Takes raw profile data from the API request and produces
    the full engineered feature vector expected by the model.

    This mirrors the exact same feature engineering done during training
    in the Jupyter notebook.
    """
    followers = raw['followers']
    follows = raw['follows']
    posts = raw['posts']
    description_length = raw['description_length']
    profile_pic = raw['profile_pic']

    # Derived features (must match notebook logic exactly)
    followers_following_ratio = (
        followers / follows if follows > 0 else 0.0
    )
    post_per_follower = (
        posts / followers if followers > 0 else 0.0
    )
    has_bio = 1 if description_length > 0 else 0
    is_active = 1 if posts > 0 else 0
    log_followers = np.log1p(followers)
    log_follows = np.log1p(follows)
    log_posts = np.log1p(posts)
    spammy_username = 1 if raw['nums_length_username'] > 0.5 else 0
    no_pic_no_bio = 1 if (profile_pic == 0 and has_bio == 0) else 0

    # Build DataFrame with columns in the exact order the model expects
    feature_names = ModelLoader.get_feature_names()

    feature_values = {
        'profile pic': profile_pic,
        'nums/length username': raw['nums_length_username'],
        'fullname words': raw['fullname_words'],
        'nums/length fullname': raw['nums_length_fullname'],
        'name==username': raw['name_eq_username'],
        'description length': description_length,
        'external URL': raw['external_url'],
        'private': raw['private'],
        '#posts': posts,
        '#followers': followers,
        '#follows': follows,
        'followers_following_ratio': followers_following_ratio,
        'post_per_follower': post_per_follower,
        'has_bio': has_bio,
        'is_active': is_active,
        'log_followers': log_followers,
        'log_follows': log_follows,
        'log_posts': log_posts,
        'spammy_username': spammy_username,
        'no_pic_no_bio': no_pic_no_bio,
    }

    df = pd.DataFrame([feature_values], columns=feature_names)
    return df


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_profile(validated_data: dict) -> dict:
    """
    Runs the full prediction pipeline:
      1. Engineer features from raw input
      2. Scale features using the saved scaler
      3. Predict using the Random Forest model
      4. Return prediction with confidence
    """
    model = ModelLoader.get_model()
    scaler = ModelLoader.get_scaler()

    # Step 1: Feature engineering
    features_df = engineer_features(validated_data)

    # Step 2: Scale
    features_scaled = scaler.transform(features_df)

    # Step 3: Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    label = "Fake" if prediction == 1 else "Real"
    confidence = round(float(probabilities[prediction]), 4)

    # Step 4: Explainable AI — per-prediction feature contributions
    xai = explain_prediction(features_scaled, features_df)

    # Extract top feature labels for the summary response
    top_features = [
        item['label'] for item in xai['feature_contributions'][:5]
    ]

    # Log to Database
    PredictionLog.objects.create(
        prediction=label,
        confidence=confidence
    )

    return {
        'prediction': label,
        'confidence_score': confidence,
        'top_features': top_features,
        'details': {
            'raw_prediction': int(prediction),
            'probability_real': round(float(probabilities[0]), 4),
            'probability_fake': round(float(probabilities[1]), 4),
        },
        'explainability': {
            'method': 'Tree Decision Path Decomposition',
            'model': 'Random Forest (100 trees)',
            'bias': xai['bias'],
            'feature_contributions': xai['feature_contributions'],
            'risk_factors': xai['risk_factors'],
            'global_feature_importance': xai['global_feature_importance'],
        },
    }

# ─── Bulk Prediction ─────────────────────────────────────────────────────────

def predict_bulk_csv(file) -> dict:
    import pandas as pd
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Could not parse CSV file: {e}")

    # Required columns that match ProfileInputSerializer schema
    required_cols = [
        'profile_pic', 'nums_length_username', 'fullname_words', 
        'nums_length_fullname', 'name_eq_username', 'description_length', 
        'external_url', 'private', 'posts', 'followers', 'follows'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    results = []
    fake_count = 0
    real_count = 0

    for index, row in df.iterrows():
        try:
            # We skip full validation here and trust the CSV data matches types
            raw_data = {col: row[col] for col in required_cols}
            
            # Predict
            pred_res = predict_profile(raw_data)
            
            # Count
            if pred_res['prediction'] == "Fake":
                fake_count += 1
            else:
                real_count += 1
                
            # Slim result for bulk output
            results.append({
                'row': index + 1,
                'prediction': pred_res['prediction'],
                'confidence': pred_res['confidence_score'],
                'top_features': pred_res['top_features']
            })
            
        except Exception as e:
            results.append({
                'row': index + 1,
                'error': str(e)
            })

    return {
        'total_processed': len(df),
        'total_fake': fake_count,
        'total_real': real_count,
        'results': results
    }

