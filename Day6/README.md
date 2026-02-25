# Day 6 — Model Governance with MLflow Registry

## Goal
Controlled promotion of ML models through **Registry stages**:
Candidate → Staging → Production → Archived.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# mac/linux:
# source .venv/bin/activate

pip install -r requirements.txt
python data/make_dataset.py