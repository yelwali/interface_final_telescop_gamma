# 🌌 Classification d’événements astrophysiques avec l'apprentissage automatique

Ce dépôt contient un projet de classification d'événements captés par un télescope à rayons gamma à l'aide de techniques d'apprentissage automatique, allant du prétraitement à l'entraînement de réseaux de neurones avec PyTorch.

---

## 📂 Contenu

### 1. `la_visualisation_preprocessing_machine_learning.ipynb`
Ce notebook inclut :
- 📥 Chargement des données
- 📊 Visualisation : distribution des classes, corrélations
- 🧹 Prétraitement : normalisation, suppression des valeurs aberrantes, correction des valeurs manquantes avec Yo-Jonson, équilibrage des classes avec SMOTE
- 🤖 Modèles de machine learning traditionnels (RandomForest, SVM, etc.)

### 2. `le_reseaux_neurones_pytorch.ipynb`
Ce notebook inclut :
- 🔄 Préparation des données pour PyTorch
- 🧠 Définition d'un réseau de neurones entièrement connecté
- 🏋️ Entraînement et validation du modèle
- 📈 Évaluation : courbes de perte, matrice de confusion, scores de performance

---

## 📊 Résultats des Modèles

### 🔬 Réseaux de Neurones (PyTorch)

| Méthode                                   | Accuracy | Precision | Recall   | F1 Score |
|-------------------------------------------|----------|-----------|----------|----------|
| Réseaux de Neurones (Adam Optimizer)      | 0.877965 | 0.878371  | 0.877965 | 0.867373 |
| Réseaux de Neurones (SGD Optimizer)       | 0.864991 | 0.866386  | 0.864991 | 0.864797 |
| Réseaux de Neurones (RandomizedSearchCV)  | 0.867423 | 0.867672  | 0.867423 | 0.867373 |
| Réseaux de Neurones (Grid Search)         | 0.877357 | 0.877608  | 0.877357 | 0.877311 |

### ⚙️ Modèles de Machine Learning Traditionnels

| Modèle                      | Accuracy | Precision | Recall   | F1 Score |
|----------------------------|----------|-----------|----------|----------|
| Logistic Regression        | 0.803966 | 0.811570  | 0.791759 | 0.801459 |
| Decision Tree              | 0.841470 | 0.879157  | 0.790866 | 0.832128 |
| Extreme Gradient Boosting  | 0.891097 | 0.908288  | 0.870011 | 0.888623 |
| Gradient Boosting          | 0.819900 | 0.817094  | 0.824439 | 0.820676 |
| Random Forest              | 0.896895 | 0.906564  | 0.885013 | 0.895418 |
| Support Vector Machine     | 0.869121 | 0.909634  | 0.819655 | 0.862262 |
| Gaussian Naive Bayes       | 0.675446 | 0.722498  | 0.674546 | 0.656012 |

---
## 🐳 Conteneurisation avec Docker

Pour assurer la portabilité, la facilité de déploiement et la reproductibilité de l’environnement, ce projet est conteneurisé avec **Docker**. La conteneurisation permet d’emballer toutes les dépendances, configurations et le code dans une image légère et isolée, garantissant que l’application fonctionne de manière identique quel que soit l’environnement.


### Exemple de fichier `requirements.txt`

```
flask==3.1.0
scikit-learn==1.6.1
xgboost==2.1.4
joblib==1.4.2
numpy==2.0.1
pandas==2.2.3
torch==2.5.1
matplotlib==3.8.4
seaborn==0.12.2
```

---

## 🐳 Conteneurisation avec Docker

### 🔧 Construction de l’image

```bash
docker build -t astrophysics-classification .
```

### 🚀 Lancement du conteneur

```bash
docker run -p 8888:8888 -v $(pwd):/app astrophysics-classification
```

### 📦 Exemple de Dockerfile

```dockerfile
# Utilisation d'une image Python officielle slim
FROM python:3.12.4-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Répertoire de travail
WORKDIR /app

# Copier uniquement le fichier des dépendances
COPY temp_requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir --retries 5 -r temp_requirements.txt

# Copier le reste des fichiers de l'application
COPY . .

# Exposer le port (ex : 5000 pour Flask)
EXPOSE 5000

# Commande de démarrage
CMD ["python", "app.py"]
```

---

## 🧾 Conclusion

Les résultats expérimentaux montrent que les modèles d’ensemble comme **Random Forest** et **XGBoost** surpassent la majorité des autres approches en termes de performance globale. **Random Forest**, en particulier, offre un excellent compromis entre précision, rappel et f1-score, ce qui en fait un choix pertinent pour les tâches de classification.

Concernant les réseaux de neurones, les performances sont également solides, surtout avec des optimisations comme **Grid Search** ou l’**optimiseur Adam**. Cependant, ils nécessitent un temps d'entraînement plus long et une configuration plus fine des hyperparamètres.

**En résumé** :

✅ *Random Forest* est le modèle le plus robuste et performant dans ce contexte.  
⚙️ *XGBoost* est très compétitif, notamment pour maximiser les performances.  
🧠 *Les réseaux de neurones* sont prometteurs, surtout pour des architectures plus complexes ou l’intégration de données non structurées à l’avenir.

Ce travail met en évidence l'importance du **choix du modèle** en fonction des **ressources disponibles**, du **besoin en interprétabilité**, de la **performance** et de la **scalabilité**.

---
