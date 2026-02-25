import pandas as pd
from sklearn.datasets import load_breast_cancer

def main():
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    df.rename(columns={"target": "label"}, inplace=True)
    out = "D:\\MLOPs\\Day6\\data\\breast_cancer.csv"
    df.to_csv(out, index=False)
    print(f"✅ Saved: {out} shape={df.shape}")

if __name__ == "__main__":
    main()