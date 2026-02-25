Day6/
├── data/
│   ├── make_dataset.py
│   └── breast_cancer.csv                  # generated
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── train_register.py                  # trains multiple models + registers in Registry
│   ├── promote.py                         # automation: candidate → staging/production
│   └── utils.py                           # shared helpers
│
├── scripts/
│   └── start_mlflow_server.ps1            # optional Windows helper
│
├── requirements.txt
└── README.md