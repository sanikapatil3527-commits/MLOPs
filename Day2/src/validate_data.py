import pandas as pd
import sys

REQUIRED_COLUMNS = [
    "Age", "Total_Purchase", "Account_Manager",
    "Years", "Num_Sites", "Location", "Company", "Churn"
]

def main():
    df = pd.read_csv("data/churn.csv")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        sys.exit(1)

    if df.isnull().sum().sum() > 0:
        print("⚠️ Warning: Missing values detected")

    print("✅ Data validation passed")

if __name__ == "__main__":
    main()
