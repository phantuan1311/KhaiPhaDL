import joblib
import os

def load_model(model_path="model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")
    return joblib.load(model_path)
