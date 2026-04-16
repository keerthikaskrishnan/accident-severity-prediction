import os
import joblib
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def load_ml_models():
    lr = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    return lr, rf, xgb

def load_dl_model():
    dl_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "dl_model.keras"))
    return dl_model

def load_transformers():
    preprocessor = joblib.load(os.path.join(DATA_DIR, "preprocessor.pkl"))
    svd = joblib.load(os.path.join(DATA_DIR, "svd_transformer.pkl"))
    scaler = joblib.load(os.path.join(DATA_DIR, "svd_scaler.pkl"))
    return preprocessor, svd, scaler
