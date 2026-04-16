import shap
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(MODEL_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

print("Loading model and data...")

# Load model + sparse test data
xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
X_test = joblib.load(os.path.join(DATA_DIR, "X_test_processed.pkl"))

# ---- Sample small background set (20 rows) ----
background_size = min(20, X_test.shape[0])
background_idx = np.random.choice(X_test.shape[0], background_size, replace=False)
X_background = X_test[background_idx]

# ---- Sample SHAP evaluation set (100 rows) ----
sample_size = min(100, X_test.shape[0])
sample_idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
X_sample = X_test[sample_idx]

print(f"Background sample: {background_size} rows")
print(f"SHAP sample: {sample_size} rows")

# ---- KernelExplainer (works with sparse matrices) ----
explainer = shap.KernelExplainer(xgb.predict, X_background)

print("Computing SHAP values (this may take a few minutes)...")
shap_values = explainer.shap_values(X_sample, nsamples=50)

print("SHAP values computed.")

# ---- SHAP Summary Plot ----
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=300)
plt.close()

print("Saved shap_summary.png")

# ---- SHAP Bar Plot ----
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_bar.png"), dpi=300)
plt.close()

print("Saved shap_bar.png")

print("SHAP plots generated successfully!")
