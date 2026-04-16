import pandas as pd
import os

# ---------------------------------------------------------
# 1. Set file paths
# ---------------------------------------------------------
DATA_DIR = "../data"

COLLISION_FILE = os.path.join(DATA_DIR, "dft-road-casualty-statistics-collision-last-5-years.csv")
VEHICLE_FILE = os.path.join(DATA_DIR, "dft-road-casualty-statistics-vehicle-last-5-years.csv")
CASUALTY_FILE = os.path.join(DATA_DIR, "dft-road-casualty-statistics-casualty-last-5-years.csv")

# ---------------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------------
print("Loading datasets...")

collisions = pd.read_csv(COLLISION_FILE)
vehicles = pd.read_csv(VEHICLE_FILE)
casualties = pd.read_csv(CASUALTY_FILE)

print("Loaded:")
print(f"  Collisions: {collisions.shape}")
print(f"  Vehicles:   {vehicles.shape}")
print(f"  Casualties: {casualties.shape}")

# ---------------------------------------------------------
# 3. Standardize column names (remove spaces, lowercase)
# ---------------------------------------------------------
collisions.columns = collisions.columns.str.strip().str.lower().str.replace(" ", "_")
vehicles.columns = vehicles.columns.str.strip().str.lower().str.replace(" ", "_")
casualties.columns = casualties.columns.str.strip().str.lower().str.replace(" ", "_")

# ---------------------------------------------------------
# 4. Check merge key (collision_index)
# ---------------------------------------------------------
if "collision_index" not in collisions.columns:
    raise KeyError("collision_index not found in collisions dataset")

if "collision_index" not in vehicles.columns:
    raise KeyError("collision_index not found in vehicles dataset")

if "collision_index" not in casualties.columns:
    raise KeyError("collision_index not found in casualties dataset")

print("\nMerge key 'collision_index' found in all datasets.")

# ---------------------------------------------------------
# 5. Merge collisions + vehicles
# ---------------------------------------------------------
print("\nMerging collisions + vehicles...")
merged_cv = collisions.merge(vehicles, on="collision_index", how="left")
print(f"Merged collisions + vehicles shape: {merged_cv.shape}")

# ---------------------------------------------------------
# 6. Merge with casualties
# ---------------------------------------------------------
print("Merging with casualties...")
merged_full = merged_cv.merge(casualties, on="collision_index", how="left")
print(f"Final merged dataset shape: {merged_full.shape}")

# ---------------------------------------------------------
# 7. Save merged dataset
# ---------------------------------------------------------
OUTPUT_FILE = os.path.join(DATA_DIR, "merged_5year_dataset.csv")
merged_full.to_csv(OUTPUT_FILE, index=False)

print(f"\nMerged dataset saved to: {OUTPUT_FILE}")
print("\nData loading and merging completed successfully.")
