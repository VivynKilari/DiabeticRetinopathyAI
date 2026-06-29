from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET = PROJECT_ROOT / "datasets"

train_df = pd.read_csv(DATASET / "train_1.csv")
val_df = pd.read_csv(DATASET / "valid.csv")
test_df = pd.read_csv(DATASET / "test.csv")

print("="*60)
print("DATASET SUMMARY")
print("="*60)

print(f"Training Images : {len(train_df)}")
print(f"Validation Images : {len(val_df)}")
print(f"Testing Images : {len(test_df)}")

print("\nTraining Classes")
print(train_df.iloc[:,1].value_counts().sort_index())

print("\nValidation Classes")
print(val_df.iloc[:,1].value_counts().sort_index())

print("\nTesting Classes")
print(test_df.iloc[:,1].value_counts().sort_index())

print(train_df.columns)
print(train_df.head())