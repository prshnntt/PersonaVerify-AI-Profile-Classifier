# 🔍 PersonaVerify — AI Fake Profile Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-4.2-092E20?style=for-the-badge&logo=django&logoColor=white)
![DRF](https://img.shields.io/badge/Django_REST_Framework-3.15-red?style=for-the-badge&logo=django&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-1.4-blue?style=for-the-badge)
![HTML/CSS/JS](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-E34F26?style=for-the-badge&logo=html5&logoColor=white)

**Detect fake social media profiles using Machine Learning — with confidence scores and explainable AI.**

[Features](#-features) · [Model Details](#-model-details) · [Run Locally](#-run-locally) · [API Usage](#-api-usage) · [Project Structure](#-project-structure)

</div>

---

## 📌 Problem Statement

Fake social media profiles are widely used for spam, misinformation, and fraud. Detecting them manually is time-consuming and inconsistent. PersonaVerify is an end-to-end Machine Learning system that automatically classifies a social media profile as **Fake or Real** using 20 engineered features — including follower ratios, username patterns, and bio signals. It returns a **confidence score** (e.g. 89% fake probability) and explains *why* the prediction was made using **Random Forest feature importances**, making the system both accurate and interpretable.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔴🟢 **Fake / Real Prediction** | Classifies any profile instantly using a trained Random Forest model |
| 📊 **Confidence Score** | Returns probability (e.g. `89.0%`) — not just a binary label |
| 🧠 **Explainable AI** | Shows top 5 features driving each prediction with importance scores |
| ⚖️ **Compare All 3 Models** | Run the same profile through all 3 classifiers to compare results |
| 📂 **Bulk CSV Upload** | Upload up to 500 profiles at once and get batch predictions |
| 📈 **Live Dashboard** | Tracks all predictions — fake/real counts, recent history |
| 🔌 **REST API** | Clean DRF-powered API — usable from any frontend or cURL |
| 🛡️ **Input Validation** | DRF serializers validate every field — returns helpful 400 errors |

---

## 🎯 Live Demo

To see the system in action, clone the repo and run it locally — the UI is served at `http://127.0.0.1:8000/` with a dark-themed dashboard, single prediction form, bulk CSV upload, and model info tab all built in.

**Try a fake profile instantly:**
```
No profile pic · Username 75% numeric · No bio · 18 followers · Following 3500 · 2 posts
→ Result: 🔴 Fake — 91.5% confidence
```

**Try a real profile:**
```
Has profile pic · Clean username · Bio 85 chars · 980 followers · Following 420 · 142 posts
→ Result: 🟢 Real — low fake probability
```

---

## 🤖 Model Details

### Dataset
- **Source:** Instagram fake/real profile dataset
- **Size:** 576 training profiles · 120 test profiles
- **Balance:** Perfectly balanced — 50% Fake, 50% Real
- **Features:** 11 raw + 9 engineered = **20 total features**

### Engineered Features (Key Ones)
| Feature | Why It Matters |
|---|---|
| `followers_following_ratio` | Fake accounts follow thousands but have few followers back |
| `log_followers` | Log-transform reduces heavy right-skew in counts |
| `no_pic_no_bio` | Combined red flag — no photo AND no bio |
| `spammy_username` | Usernames with >30% numeric chars (e.g. `user19283746`) |
| `post_per_follower` | Activity signal relative to audience size |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 93.33% | 90.62% | 96.67% | 93.55% | 98.33% |
| Decision Tree | 87.50% | 85.71% | 90.00% | 87.80% | 90.11% |
| **Random Forest ✅** | **94.17%** | **92.06%** | **96.67%** | **94.31%** | **99.51%** |

**Why Random Forest was selected:**
- Highest accuracy (94.17%) and AUC-ROC (99.51%)
- Ensemble of 200 decision trees — reduces variance vs single tree
- Naturally provides feature importances for Explainable AI
- Robust to outliers and does not require feature scaling (though we scale for consistency with LR)

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/kushhcodes/PersonaVerify-AI-Profile-Classifier.git
cd PersonaVerify-AI-Profile-Classifier

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and set your SECRET_KEY (or leave the default for local dev)

# 5. Apply database migrations
python manage.py makemigrations api
python manage.py migrate

# 6. Run the development server
python manage.py runserver
```

**Open your browser:**
- 🌐 Frontend UI → http://127.0.0.1:8000/
- 🔌 API Root    → http://127.0.0.1:8000/api/
- ⚙️ Admin Panel → http://127.0.0.1:8000/admin/

---

## 🔌 API Usage

### `POST /api/predict/` — Single Profile Prediction

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "profile_pic": 0,
    "nums_length_username": 0.75,
    "fullname_words": 1,
    "nums_length_fullname": 0,
    "name_equals_username": 1,
    "description_length": 0,
    "external_url": 0,
    "private": 0,
    "posts_count": 2,
    "followers_count": 18,
    "following_count": 3500
  }'
```

**Response:**
```json
{
  "prediction": "Fake",
  "is_fake": true,
  "confidence_score": 0.915,
  "confidence_pct": "91.5%",
  "real_probability": 0.085,
  "label_color": "red",
  "top_features": ["log_followers", "followers_following_ratio", "log_follows", "no_pic_no_bio", "spammy_username"],
  "feature_insights": [
    "18 followers",
    "Follower/following ratio: 0.01 — low — suspicious",
    "Log-scaled following: 8.16",
    "No profile picture AND no bio (double red flag)",
    "Spammy numeric username (red flag)"
  ],
  "feature_scores": {
    "log_followers": 0.1823,
    "followers_following_ratio": 0.1541,
    "log_follows": 0.1102,
    "no_pic_no_bio": 0.0891,
    "spammy_username": 0.0743
  }
}
```

---

### `GET /api/dashboard/` — Live Stats

```bash
curl http://127.0.0.1:8000/api/dashboard/
```

**Response:**
```json
{
  "total_predictions": 42,
  "total_fake": 27,
  "total_real": 15,
  "fake_percentage": 64.3,
  "real_percentage": 35.7,
  "recent_predictions": [...]
}
```

---

### `GET /api/model-info/` — Model Accuracy & Feature Importances

```bash
curl http://127.0.0.1:8000/api/model-info/
```

---

### `POST /api/compare-models/` — Run All 3 Models

Uses the same request body as `/api/predict/`. Returns predictions from all 3 models with a consensus.

---

### `POST /api/bulk-predict/` — CSV Bulk Upload

```bash
curl -X POST http://127.0.0.1:8000/api/bulk-predict/ \
  -F "file=@test.csv"
```

---

### All Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict/` | Single profile prediction |
| `POST` | `/api/bulk-predict/` | CSV upload — batch predictions |
| `GET` | `/api/dashboard/` | Aggregate stats and history |
| `GET` | `/api/model-info/` | Model accuracy + feature importances |
| `GET` | `/api/feature-info/` | Feature descriptions |
| `POST` | `/api/compare-models/` | Compare all 3 models on same input |

---

## 📁 Project Structure

```
PersonaVerify-AI-Profile-Classifier/
│
├── manage.py                        # Django entry point
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (SECRET_KEY, DEBUG)
├── .env.example                     # Template for .env — safe to commit
│
├── backend/                         # Django project config
│   ├── settings.py                  # All settings — reads from .env
│   ├── urls.py                      # Main URL router
│   └── wsgi.py                      # WSGI server entry point
│
├── api/                             # Core Django app
│   ├── ml_engine.py                 # ML singleton — loads model at startup
│   ├── views.py                     # All 6 API endpoint views
│   ├── serializers.py               # DRF input validation + output formatting
│   ├── models.py                    # PredictionLog + BulkUploadJob (SQLite)
│   ├── urls.py                      # API URL patterns
│   ├── apps.py                      # AppConfig — pre-loads ML model
│   ├── admin.py                     # Django admin registrations
│   └── frontend_urls.py             # Serves HTML frontend at /
│
├── ml_model/                        # Saved Joblib artifacts (from notebook)
│   ├── fake_profile_model.pkl       # Best model — Random Forest
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── scaler.pkl                   # Fitted StandardScaler
│   ├── feature_names.pkl            # Ordered list of 20 feature names
│   ├── feature_importances.pkl      # DataFrame of RF importances
│   └── model_metrics.pkl            # Accuracy/precision/recall for all 3 models
│
├── templates/
│   └── index.html                   # Full frontend UI (served by Django)
│
├── static/
│   ├── css/style.css                # Dark-theme stylesheet
│   └── js/app.js                    # Frontend logic (Fetch API calls)
│
├── media/
│   └── uploads/                     # Uploaded CSVs land here
│
├── tests/                           # Unit tests
│   ├── __init__.py
│   └── test_api.py                  # 8 test cases for all endpoints
│
├── conftest.py                      # Shared test fixtures and config
│
└── docs/
    └── screenshots/                 # Add your screenshots here
        ├── fake_prediction.png
        ├── real_prediction.png
        ├── model_comparison.png
        ├── dashboard.png
        └── model_info.png
```

---

## 🧪 Running Tests

```bash
# Run all tests
python manage.py test tests

# Run with verbosity (see each test name)
python manage.py test tests --verbosity=2

# Run a specific test class
python manage.py test tests.test_api.PredictEndpointTests
```

**Test Coverage:**
- ✅ Valid fake profile prediction → returns 200 + correct keys
- ✅ Valid real profile prediction → returns 200 + "Real"
- ✅ Missing required field → returns 400
- ✅ Invalid field type (string instead of int) → returns 400
- ✅ Edge case — zero followers/following/posts → returns 200
- ✅ Dashboard endpoint → returns stats keys
- ✅ Model info endpoint → returns metrics + feature importances
- ✅ Compare models → returns 3 model results + consensus

---

## 🔮 Future Improvements

- [ ] **SHAP Values** — Add SHAP (SHapley Additive exPlanations) for more granular per-prediction explanations
- [ ] **User Authentication** — JWT-based auth so users can track their own prediction history
- [ ] **Download Results** — Export bulk prediction results as CSV
- [ ] **Rate Limiting** — Per-user rate limiting using DRF throttling with Redis backend
- [ ] **PostgreSQL** — Migrate from SQLite to PostgreSQL for production
- [ ] **Docker** — Containerize with Docker + docker-compose for one-command deployment
- [ ] **CI/CD** — GitHub Actions pipeline to run tests on every push
- [ ] **Model Retraining** — Admin endpoint to retrain the model on new data
- [ ] **React Frontend** — Upgrade from vanilla JS to a React SPA

---

## 🧠 Key Technical Decisions

**Q: Why use `AppConfig.ready()` to load the model?**
Loading a 1.4MB `.pkl` file inside a view would add ~200ms to every single request. By loading it once at startup in `ready()`, the model stays in RAM — each prediction becomes a sub-10ms matrix operation.

**Q: Why DRF Serializers instead of manual validation?**
Serializers provide automatic field-level validation, type coercion, and standardized 400 responses. A view should never need `if 'field' not in request.data`.

**Q: Why log predictions to a database?**
The dashboard needs aggregate stats without re-running the model. Every `POST /api/predict/` writes one row to `PredictionLog` — the dashboard just does `SELECT COUNT(*) WHERE is_fake=True`.

---

## 👤 Author

**Kush** — [@kushhcodes](https://github.com/kushhcodes)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).# PersonaVerify-AI-Profile-Classifier
