import os
import json
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DATA_DIR = "../data"
MODEL_DIR = "../models"
FIG_DIR = "../models/figures"

# Create folder for figures if not exists
os.makedirs(FIG_DIR, exist_ok=True)

print("Loading deep learning model and data...")

# ------------------------------
# Load DL model + data + scaler
# ------------------------------
dl_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "dl_model.keras"))

X_test = joblib.load(os.path.join(DATA_DIR, "X_test_reduced.pkl"))
y_test = joblib.load(os.path.join(DATA_DIR, "y_test_reduced.pkl"))
scaler = joblib.load(os.path.join(DATA_DIR, "svd_scaler.pkl"))

X_test_scaled = scaler.transform(X_test)

# ------------------------------
# DL Predictions
# ------------------------------
print("Running DL predictions...")
y_pred_probs = dl_model.predict(X_test_scaled)
y_pred_dl = np.argmax(y_pred_probs, axis=1)

dl_accuracy = accuracy_score(y_test, y_pred_dl)
print("\nDeep Learning Test Accuracy:", dl_accuracy)

print("\nDeep Learning Classification Report:")
print(classification_report(y_test, y_pred_dl, digits=4))

# ------------------------------
# Confusion Matrix
# ------------------------------
cm = confusion_matrix(y_test, y_pred_dl)
labels = ["Slight", "Serious", "Fatal"]

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Deep Learning Model")
plt.tight_layout()

# SAVE FIGURE
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_dl.png"), dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------
# Training Curves
# ------------------------------
print("\nPlotting training curves...")

with open(os.path.join(MODEL_DIR, "dl_history.json"), "r") as f:
    history = json.load(f)

# Accuracy curve
plt.figure(figsize=(10, 4))
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Val Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# SAVE FIGURE
plt.savefig(os.path.join(FIG_DIR, "training_accuracy_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

# Loss curve
plt.figure(figsize=(10, 4))
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# SAVE FIGURE
plt.savefig(os.path.join(FIG_DIR, "training_loss_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------
# ML vs DL Comparison
# ------------------------------
print("\nLoading classical ML models...")

lr = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))

X_test_processed = joblib.load(os.path.join(DATA_DIR, "X_test_processed.pkl"))
y_test_original = joblib.load(os.path.join(DATA_DIR, "y_test.pkl")) - 1

acc_lr = accuracy_score(y_test_original, lr.predict(X_test_processed))
acc_rf = accuracy_score(y_test_original, rf.predict(X_test_processed))
acc_xgb = accuracy_score(y_test_original, xgb.predict(X_test_processed))

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Deep Learning"],
    "Accuracy": [acc_lr, acc_rf, acc_xgb, dl_accuracy]
})

print("\nModel Comparison:")
print(comparison_df)

comparison_df.to_csv(os.path.join(MODEL_DIR, "ml_vs_dl_comparison.csv"), index=False)
