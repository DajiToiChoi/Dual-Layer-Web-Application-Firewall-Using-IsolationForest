from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
DB_PATH = BASE_DIR / "waf_logs.db"

MODELS_DIR = BASE_DIR / "models"
L1_MODEL_PATH = MODELS_DIR / "l1_iforest.joblib"
L2_MODEL_PATH = MODELS_DIR / "l2_svm_tfidf.joblib"

# API Key demo (can be moved to DB later)
DEMO_API_KEYS = {"SECRET_DEMO_KEY_123"}

# Rate limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW_SECONDS = 60

# L1 thresholding: IsolationForest predicts {-1 anomaly, 1 normal}
# We also compute a continuous anomaly score in [0,1] and may alert/log it.
ANOMALY_BLOCK_ON_L2 = True  # if True: L1 anomaly -> run L2 to decide block
