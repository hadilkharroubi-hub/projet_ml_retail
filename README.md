# Projet ML Retail — Analyse Comportementale Clientèle

## Description
Analyse comportementale d'une clientèle e-commerce de cadeaux.
Chaîne complète : Exploration → Préparation → Modélisation → Évaluation → Déploiement.

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/hadilkharroubi-hub/projet_ml_retail.git
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Structure du projet
```
projet_ml_retail/
├── data/
│   ├── raw/           # Données brutes originales
│   ├── processed/     # Données nettoyées
│   └── train_test/    # Données splittées (train/test)
├── notebooks/         # Notebooks Jupyter (prototypage)
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
├── models/            # Modèles sauvegardés (.pkl, .joblib)
├── app/               # Application Flask
├── reports/           # Rapports et visualisations
├── requirements.txt
├── README.md
└── .gitignore
```

## Utilisation

### Lancer le preprocessing
```bash
python src/preprocessing.py
```

### Lancer l'exploration (notebook)
```bash
jupyter notebook notebooks/
```