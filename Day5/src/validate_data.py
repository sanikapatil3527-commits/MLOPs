import pandas as pd
from src.config import SETTINGS

REQUIRED_MIN_ROWS = 100
MAX_MISSING_RATIO = 0.01  # 1%

def main():
    df = pd.read_csv(SETTINGS.data_path)

    # 1) basic checks
    if df.shape[0] < REQUIRED_MIN_ROWS:
        raise ValueError(f"Not enough rows: {df.shape[0]} < {REQUIRED_MIN_ROWS}")

    if SETTINGS.target_col not in df.columns:
        raise ValueError(f"Missing target col '{SETTINGS.target_col}' in {list(df.columns)}")

    # 2) missing values
    missing_ratio = df.isna().mean().max()
    if missing_ratio > MAX_MISSING_RATIO:
        raise ValueError(f"Too many missing values: max_missing_ratio={missing_ratio:.3f} > {MAX_MISSING_RATIO}")

    # 3) label checks
    y = df[SETTINGS.target_col]
    unique = sorted(y.unique().tolist())
    if not set(unique).issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1. Found: {unique}")

    # 4) feature sanity
    X = df.drop(columns=[SETTINGS.target_col])
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric features found (unexpected for this dataset): {non_numeric}")

    print("✅ Data validation passed.")
    print(f"   rows={df.shape[0]}, cols={df.shape[1]}, max_missing_ratio={missing_ratio:.4f}, labels={unique}")

if __name__ == "__main__":
    main()
