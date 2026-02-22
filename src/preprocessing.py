import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from utils import (load_data, fix_support_tickets, fix_satisfaction,
                   parse_registration_date, parse_last_login_ip,
                   drop_useless_features, feature_engineering,
                   impute_missing, missing_report)

# ============================================================
# 1. CHARGEMENT
# ============================================================
df = load_data('../data/raw/ton_fichier.csv')

# ============================================================
# 2. NETTOYAGE
# ============================================================
df = drop_useless_features(df)
df = fix_support_tickets(df)
df = fix_satisfaction(df)
df = parse_registration_date(df)
df = parse_last_login_ip(df)

# ============================================================
# 3. IMPUTATION DES VALEURS MANQUANTES
# ============================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Retirer la target de la liste !
if 'Churn' in numeric_cols:
    numeric_cols.remove('Churn')

df = impute_missing(df, numeric_cols, strategy='median')

# Imputation catégorielle (mode)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
df = feature_engineering(df)

# ============================================================
# 5. ENCODAGE DES VARIABLES CATÉGORIELLES
# ============================================================
# Encodage ordinal
ordinal_mappings = {
    'AgeCategory':  ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
    'SpendingCat':  ['Low', 'Medium', 'High', 'VIP'],
    'LoyaltyLevel': ['Nouveau', 'Jeune', 'Établi', 'Ancien', 'Inconnu'],
    'ChurnRisk':    ['Faible', 'Moyen', 'Élevé', 'Critique'],
    'BasketSize':   ['Petit', 'Moyen', 'Grand', 'Inconnu'],
    'PreferredTime':['Matin', 'Midi', 'Après-midi', 'Soir', 'Nuit'],
}

for col, categories in ordinal_mappings.items():
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=categories, ordered=True).codes

# One-Hot encoding
onehot_cols = ['CustomerType', 'FavoriteSeason', 'Region',
               'WeekendPref', 'ProdDiversity', 'Gender',
               'AccountStatus', 'RFMSegment']
onehot_cols = [c for c in onehot_cols if c in df.columns]
df = pd.get_dummies(df, columns=onehot_cols, drop_first=False)

# Target encoding pour Country (fréquence)
if 'Country' in df.columns:
    country_freq = df['Country'].value_counts(normalize=True)
    df['Country'] = df['Country'].map(country_freq)

# Supprimer CustomerID (identifiant sans valeur prédictive)
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# ============================================================
# 6. SAUVEGARDE DES DONNÉES NETTOYÉES
# ============================================================
df.to_csv('../data/processed/data_cleaned.csv', index=False)
print("✅ Données nettoyées sauvegardées dans data/processed/data_cleaned.csv")

# ============================================================
# 7. SÉPARATION TRAIN/TEST
# ============================================================
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 8. NORMALISATION (fit sur train uniquement !)
# ============================================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

# ============================================================
# 9. SAUVEGARDE TRAIN/TEST
# ============================================================
X_train_scaled.to_csv('../data/train_test/X_train.csv', index=False)
X_test_scaled.to_csv('../data/train_test/X_test.csv',   index=False)
y_train.to_csv('../data/train_test/y_train.csv',         index=False)
y_test.to_csv('../data/train_test/y_test.csv',           index=False)

import joblib
joblib.dump(scaler, '../models/scaler.joblib')

print("✅ Train/Test sauvegardés dans data/train_test/")
print(f"   X_train: {X_train_scaled.shape} | X_test: {X_test_scaled.shape}")
print(f"   Distribution Churn train: {y_train.value_counts().to_dict()}")