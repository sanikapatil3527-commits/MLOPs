# 🧭 MLOps Standards & Principles

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

