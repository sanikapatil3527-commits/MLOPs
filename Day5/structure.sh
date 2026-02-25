Day5/
├── data/
│   ├── make_dataset.py
│   └── breast_cancer.csv          # generated
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── validate_data.py           # Step 1
│   ├── train.py                   # Step 2
│   ├── evaluate.py                # Step 3
│   └── gates.py                   # Step 4
├── .github/
│   └── workflows/
│       └── ci_ml.yml              # Step 5
├── mlruns/                        # local MLflow tracking store
├── requirements.txt
└── README.md
