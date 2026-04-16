import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# 1. Load processed data
# ---------------------------------------------------------
DATA_DIR = "../data"
MODEL_DIR = "../models"

X_train = joblib.load(os.path.join(DATA_DIR, "X_train_processed.pkl"))
X_test = joblib.load(os.path.join(DATA_DIR, "X_test_processed.pkl"))
y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
y_test = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))

print("Loaded processed datasets.")
print("Train:", X_train.shape, "Test:", X_test.shape)

# ---------------------------------------------------------
# FIX: Remap labels for XGBoost (1,2,3 → 0,1,2)
# ---------------------------------------------------------
y_train = y_train - 1
y_test = y_test - 1

# ---------------------------------------------------------
# 2. Helper: Evaluate model
# ---------------------------------------------------------
def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test, preds, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }

# ---------------------------------------------------------
# 3. Train Logistic Regression (fixed)
# ---------------------------------------------------------
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(
    solver="saga",        # best for sparse high-dimensional data
    max_iter=50,          # keeps training time reasonable
    penalty="l2"
)
log_reg.fit(X_train, y_train)
joblib.dump(log_reg, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

# ---------------------------------------------------------
# 4. Train Random Forest
# ---------------------------------------------------------
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train, y_train)
joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))

# ---------------------------------------------------------
# 5. Train XGBoost (fixed)
# ---------------------------------------------------------
print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(np.unique(y_train)),
    n_jobs=-1,
    tree_method="hist"
)
xgb.fit(X_train, y_train)
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgboost.pkl"))

# ---------------------------------------------------------
# 6. Evaluate all models
# ---------------------------------------------------------
results = []
results.append(evaluate(log_reg, X_test, y_test, "Logistic Regression"))
results.append(evaluate(rf, X_test, y_test, "Random Forest"))
results.append(evaluate(xgb, X_test, y_test, "XGBoost"))

# ---------------------------------------------------------
# 7. Print comparison table
# ---------------------------------------------------------
print("\n===== MODEL PERFORMANCE SUMMARY =====")
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv(os.path.join(MODEL_DIR, "model_performance_summary.csv"), index=False)

print("\nTraining complete. Models saved in /models folder.")
