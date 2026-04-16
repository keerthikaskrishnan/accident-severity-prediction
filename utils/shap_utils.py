import os
import joblib
import shap
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
FIG_DIR = os.path.join(MODEL_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def get_xgb_and_data():
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    X_test = joblib.load(os.path.join(BASE_DIR, "..", "data", "X_test_processed.pkl"))
    return xgb, X_test

def compute_global_shap():
    xgb, X_test = get_xgb_and_data()
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(os.path.join(FIG_DIR, "shap_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

def get_local_shap(index=0):
    xgb, X_test = get_xgb_and_data()
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values, X_test, index
