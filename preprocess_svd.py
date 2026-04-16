import os
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

DATA_DIR = "../data"

print("Loading sparse matrices...")
X_train = joblib.load(os.path.join(DATA_DIR, "X_train_processed.pkl"))
X_test = joblib.load(os.path.join(DATA_DIR, "X_test_processed.pkl"))
y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
y_test = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))

# Ensure labels are 0, 1, 2
y_train = y_train - 1
y_test = y_test - 1

print("Running optimized TruncatedSVD (100 components, randomized)...")

svd = TruncatedSVD(
    n_components=100,
    algorithm="randomized",
    n_iter=5,
    random_state=42
)

X_train_reduced = svd.fit_transform(X_train)
X_test_reduced = svd.transform(X_test)

print("Saving reduced datasets...")
joblib.dump(X_train_reduced, os.path.join(DATA_DIR, "X_train_reduced.pkl"))
joblib.dump(X_test_reduced, os.path.join(DATA_DIR, "X_test_reduced.pkl"))
joblib.dump(y_train, os.path.join(DATA_DIR, "y_train_reduced.pkl"))
joblib.dump(y_test, os.path.join(DATA_DIR, "y_test_reduced.pkl"))
joblib.dump(svd, os.path.join(DATA_DIR, "svd_transformer.pkl"))

print("SVD preprocessing complete.")
