# 🌐 MLflow – Core Concepts Overview

MLflow est une plateforme open-source permettant de gérer **le cycle de vie complet du Machine Learning**, de l'expérimentation à la mise en production.

---

## 🧩 Key Characteristics

### 🌍 Language Agnostic
MLflow peut être utilisé avec :
- n’importe quelle librairie ML
- n’importe quel langage de programmation

✅ Approche **API-first et modulaire**

---

### 🔗 Compatibility

Compatible avec de nombreuses librairies ML :

- TensorFlow
- PyTorch
- Keras
- Scikit-learn
- Apache Spark
- XGBoost
- LightGBM

👉 MLflow agit comme une **couche d’intégration universelle**

---

### 🚀 Integration

MLflow permet de :
- mettre un modèle en production
- l'encapsuler dans :
  - Docker containers
  - Kubernetes clusters
  - Apache Spark jobs
  - REST APIs

🎯 Objectif : déploiement standardisé et reproductible

---

### 🏗 Creation

- Créé par **Databricks**
- Première version : **juin 2018**

MLflow est aujourd’hui un pilier du MLOps moderne.

---

## 🏛 MLflow Components

MLflow repose sur **4 modules principaux** :

---

### 📊 Tracking

> Suivre les expériences et comparer facilement

Permet :
- enregistrement des paramètres
- métriques
- artefacts
- visualisation et comparaison d'expériences

Outils :
- UI MLflow
- APIs Python / CLI

---

### 📦 Projects

> Standardiser et packager le code ML

Objectifs :
- réutilisabilité
- reproductibilité

Inclut :
- définition d’environnement
- dépendances
- structure d’exécution

---

### 🤖 Models

> Format standard de packaging de modèles

Permet :
- export uniforme
- déploiement multi-backend

Supporte :
- Docker
- Spark
- ONNX
- REST API serving

---

### 🗃 Registry

> Stockage centralisé et versioning de modèles

Fonctionnalités :
- versioning
- transition de stages (`Staging`, `Production`)
- annotations
- validation

🎯 Point central du MLOps automatisé

---

## ✅ Vue Synthétique

| Composant | Rôle |
|---|---|
| Tracking | Suivi des expériences |
| Projects | Packaging du code |
| Models | Format standard modèles |
| Registry | Versioning & gestion modèles |

---

## 🧠 Message clé

> MLflow n’est pas seulement un tracker, c’est une **colonne vertébrale MLOps** reliant entraînement, packaging, déploiement et versioning.

---
