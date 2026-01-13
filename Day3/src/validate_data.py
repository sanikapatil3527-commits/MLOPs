import sys
import pandas as pd
from src.config import SETTINGS

REQUIRED_COLUMNS = [
    "Age", "Total_Purchase", "Account_Manager",
    "Years", "Num_Sites", "Location", "Company", SETTINGS.target_col
]

def main():
    df = pd.read_csv(SETTINGS.data_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        sys.exit(1)

    if df[SETTINGS.target_col].isnull().any():
        print("❌ Target column contains missing values.")
        sys.exit(1)

    # Soft warnings (don’t fail, but notify)
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        print(f"⚠️ Warning: dataset contains {nulls} missing values (will be imputed).")

    print("✅ Data validation passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
