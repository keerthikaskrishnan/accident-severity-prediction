import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------------------------------------
# 1. Load merged dataset
# ---------------------------------------------------------
DATA_DIR = "../data"
INPUT_FILE = os.path.join(DATA_DIR, "merged_5year_dataset.csv")

print("Loading merged dataset...")
df = pd.read_csv(INPUT_FILE)
print("Loaded:", df.shape)

# ---------------------------------------------------------
# 2. Drop irrelevant / leaky / duplicate columns
# ---------------------------------------------------------
drop_cols = [
    "collision_ref_no",
    "local_authority_highway_current",
    "junction_detail_historic",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic",
    "carriageway_hazards_historic",
    "vehicle_manoeuvre_historic",
    "journey_purpose_of_driver_historic",
    "collision_injury_based",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight",
    "casualty_injury_based",
    "casualty_adjusted_severity_serious",
    "casualty_adjusted_severity_slight",
]

df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ---------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["is_weekend"] = df["date"].dt.dayofweek >= 5

# Extract hour from time
def extract_hour(t):
    try:
        return int(str(t).split(":")[0])
    except:
        return np.nan

df["hour"] = df["time"].apply(extract_hour)

# Time of day bucket
def time_bucket(h):
    if pd.isna(h):
        return "unknown"
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 21:
        return "evening"
    return "night"

df["time_of_day"] = df["hour"].apply(time_bucket)

# ---------------------------------------------------------
# 4. Handle missing values
# ---------------------------------------------------------
# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Fill numeric missing values with median
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with "unknown"
for col in categorical_cols:
    df[col] = df[col].astype(str).fillna("unknown")


# ---------------------------------------------------------
# 5. Define target and features
# ---------------------------------------------------------
TARGET = "collision_severity"

if TARGET not in df.columns:
    raise KeyError(f"Target column '{TARGET}' not found.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ---------------------------------------------------------
# 6. Preprocessing Pipeline
# ---------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# ---------------------------------------------------------
# 7. Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train/Test shapes:", X_train.shape, X_test.shape)

# ---------------------------------------------------------
# 8. Fit preprocessing pipeline
# ---------------------------------------------------------
print("Fitting preprocessing pipeline...")
preprocessor.fit(X_train)

# Transform data
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed shapes:", X_train_processed.shape, X_test_processed.shape)

# ---------------------------------------------------------
# 9. Save processed data + pipeline
# ---------------------------------------------------------
OUTPUT_TRAIN_X = os.path.join(DATA_DIR, "X_train_processed.pkl")
OUTPUT_TEST_X = os.path.join(DATA_DIR, "X_test_processed.pkl")
OUTPUT_TRAIN_Y = os.path.join(DATA_DIR, "y_train.pkl")
OUTPUT_TEST_Y = os.path.join(DATA_DIR, "y_test.pkl")
OUTPUT_PIPELINE = os.path.join(DATA_DIR, "preprocessor.pkl")

joblib.dump(X_train_processed, OUTPUT_TRAIN_X)
joblib.dump(X_test_processed, OUTPUT_TEST_X)
joblib.dump(y_train, OUTPUT_TRAIN_Y)
joblib.dump(y_test, OUTPUT_TEST_Y)
joblib.dump(preprocessor, OUTPUT_PIPELINE)

print("\nPreprocessing complete.")
print("Saved:")
print(" - X_train_processed.pkl")
print(" - X_test_processed.pkl")
print(" - y_train.pkl")
print(" - y_test.pkl")
print(" - preprocessor.pkl")
