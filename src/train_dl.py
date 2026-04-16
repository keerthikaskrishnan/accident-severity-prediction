import os
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = "../data"
MODEL_DIR = "../models"

print("Loading reduced datasets...")
X_train = joblib.load(os.path.join(DATA_DIR, "X_train_reduced.pkl"))
X_test = joblib.load(os.path.join(DATA_DIR, "X_test_reduced.pkl"))
y_train = joblib.load(os.path.join(DATA_DIR, "y_train_reduced.pkl"))
y_test = joblib.load(os.path.join(DATA_DIR, "y_test_reduced.pkl"))

# Normalize SVD output
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(DATA_DIR, "svd_scaler.pkl"))

input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))

print("Input dimension:", input_dim)
print("Classes:", num_classes)

# ------------------------------
# Build Faster Deep Learning Model
# ------------------------------
inputs = layers.Input(shape=(input_dim,))

x = layers.Dense(128, activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------
# Callbacks
# ------------------------------
early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1
)

# ------------------------------
# Train Model (Much Faster)
# ------------------------------
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=128,
    callbacks=[early_stop, reduce_lr]
)

# ------------------------------
# Evaluate
# ------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_acc)}

print("Test accuracy:", test_acc)

# ------------------------------
# Save Model + History + Metrics
# ------------------------------
model.save(os.path.join(MODEL_DIR, "dl_model.keras"))

with open(os.path.join(MODEL_DIR, "dl_history.json"), "w") as f:
    json.dump(history.history, f)

with open(os.path.join(MODEL_DIR, "dl_metrics.json"), "w") as f:
    json.dump(metrics, f)

print("Deep learning model saved.")
