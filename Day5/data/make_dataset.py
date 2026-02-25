import pandas as pd
from sklearn.datasets import load_breast_cancer

def main():
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()

    # rename target to be explicit
    # sklearn target: 0=malignant, 1=benign
    df.rename(columns={"target": "label"}, inplace=True)

    out = "data/breast_cancer.csv"
    df.to_csv(out, index=False)
    print(f"✅ Saved dataset to {out} with shape={df.shape}")

if __name__ == "__main__":
    main()
