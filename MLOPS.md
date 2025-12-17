<!-- # 🧭 MLOps Standards & Principles

> Objectif : réduire la friction entre développement et production, garantir la qualité, la traçabilité et la reproductibilité des modèles ML.

---

## 🔁 Transition Friction

Réduire les obstacles entre :
- recherche → développement
- développement → production

✅ Bonnes pratiques :
- Utiliser des **templates de notebooks** standardisés :
  - connexion base de données
  - chargement des données
  - pré-processing commun
  - structure d’expérience
- Documentation claire et à jour

👉 Résultat attendu :
- onboarding rapide
- cohérence entre équipes
- reproductibilité

---

## 🗂 Version Control System (VCS)

> "Ce qui n’est pas versionné n’existe pas."

À versionner :
- ✅ code
- ✅ données (ou leur source)
- ✅ environnements
- ✅ artefacts (modèles, métriques, figures)

Outils recommandés :
- Git / GitHub / GitLab
- DVC
- MLflow Artifacts

---

## 🚀 Performance

Objectif : exécuter les pipelines de façon efficace et scalable

Approches :
- computing distribué
- containerisation

Outils :
- Docker
- Kubernetes
- Spark
- Ray

---

## 🤖 Automation

> MLOps est **pipeline-centric**, pas model-centric

Objectifs :
- automatiser le passage du data au modèle en production
- CI/CD & CI/ML & CD/ML

Inclut :
- automatisation du training
- automatisation du déploiement
- automatisation de l'évaluation

Outils :
- GitHub Actions
- GitLab CI
- Jenkins
- MLflow Pipelines

---

## 📈 Monitoring

> Un modèle en production sans monitoring = bombe à retardement

À monitorer :
- données entrantes
- distribution des features (drift)
- latence
- uptime
- mémoire utilisée
- performance modèle

Outils recommandés :
- Prometheus
- Grafana
- MLflow Model Monitoring
- Evidently AI

---

## 🔄 Continuous Training (CT)

Automatiser :
- le retraining régulier
- ou basé sur triggers :
  - drift détecté
  - nouveau dataset
  - seuil qualité dépassé

Pipeline typique :
Data → Validation → Training → Evaluation → Registry → Deployment
↑ ↓
Monitoring ←----------←

---

## ✅ Résumé visuel

| Principe | Objectif | Outils |
|---|---|---|
| Transition Friction | Standardisation | Templates, Docs |
| Version Control | Traçabilité | Git, DVC, MLflow |
| Performance | Scalabilité | Docker, K8s |
| Automation | Pipelines | CI/CD, MLflow |
| Monitoring | Qualité prod | Prometheus, Grafana |
| Continuous Training | Adaptation | Auto retraining |

---

## 🧠 Message clé pour le cours

> Le cœur du MLOps moderne (et de MLflow) n’est pas seulement d’entraîner un modèle, mais de maintenir **un pipeline vivant, monitoré et automatisé**.

--- -->

# 🧭 MLOps Standards & Principles

> Objective: reduce friction between development and production, while ensuring model quality, traceability, and reproducibility.

---

## 🔁 Transition Friction

Reduce barriers between:
- research → development
- development → production

✅ Best practices:
- Use **standardized notebook templates**:
  - database connection
  - data loading
  - shared preprocessing
  - experiment structure
- Clear and up-to-date documentation

👉 Expected outcomes:
- faster onboarding
- team consistency
- reproducibility

---

## 🗂 Version Control System (VCS)

> "If it is not versioned, it does not exist."

What should be versioned:
- ✅ code
- ✅ data (or data sources)
- ✅ environments
- ✅ artifacts (models, metrics, figures)

Recommended tools:
- Git / GitHub / GitLab
- DVC
- MLflow Artifacts

---

## 🚀 Performance

Objective: run pipelines efficiently and at scale

Approaches:
- distributed computing
- containerization

Tools:
- Docker
- Kubernetes
- Spark
- Ray

---

## 🤖 Automation

> MLOps is **pipeline-centric**, not model-centric

Objectives:
- automate the path from data to production models
- CI/CD, CI/ML, and CD/ML

Includes:
- automated training
- automated deployment
- automated evaluation

Tools:
- GitHub Actions
- GitLab CI
- Jenkins
- MLflow Pipelines

---

## 📈 Monitoring

> A production model without monitoring is a ticking time bomb

What to monitor:
- incoming data
- feature distributions (data drift)
- latency
- uptime
- memory usage
- model performance

Recommended tools:
- Prometheus
- Grafana
- MLflow Model Monitoring
- Evidently AI

---

## 🔄 Continuous Training (CT)

Automate:
- periodic retraining
- or trigger-based retraining:
  - detected drift
  - new datasets
  - quality thresholds exceeded

Typical pipeline:

Data → Validation → Training → Evaluation → Registry → Deployment
↑ ↓
Monitoring ←-----------------------------←
---

## ✅ Visual Summary

| Principle | Objective | Tools |
|---|---|---|
| Transition Friction | Standardization | Templates, Docs |
| Version Control | Traceability | Git, DVC, MLflow |
| Performance | Scalability | Docker, Kubernetes |
| Automation | Pipelines | CI/CD, MLflow |
| Monitoring | Production Quality | Prometheus, Grafana |
| Continuous Training | Adaptation | Auto-retraining |

---

## 🧠 Key Message for the Course

> The core of modern MLOps (and MLflow) is not just training a model,  
> but maintaining **a living, monitored, and automated pipeline**.

---
