import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import (load_data, fix_support_tickets, fix_satisfaction,
                   parse_registration_date, parse_last_login_ip,
                   drop_useless_features, feature_engineering,
                   impute_missing, missing_report)

# ============================================================
# 1. CHARGEMENT
# ============================================================
df = load_data('../data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')

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
if 'Churn' in numeric_cols:
    numeric_cols.remove('Churn')

df = impute_missing(df, numeric_cols, strategy='median')

# Imputation catégorielle (mode)
cat_cols = df.select_dtypes(include=['str']).columns.tolist()
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
df = feature_engineering(df)

# ============================================================
# 5. ENCODAGE DES VARIABLES CATÉGORIELLES
# ============================================================

# --- Encodage ordinal ---
ordinal_mappings = {
    'AgeCategory':        ['18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Inconnu'],
    'SpendingCategory':   ['Low', 'Medium', 'High', 'VIP'],
    'LoyaltyLevel':       ['Nouveau', 'Jeune', 'Établi', 'Ancien', 'Inconnu'],
    'ChurnRiskCategory':  ['Faible', 'Moyen', 'Élevé', 'Critique'],
    'BasketSizeCategory': ['Petit', 'Moyen', 'Grand', 'Inconnu'],
    'PreferredTimeOfDay': ['Matin', 'Midi', 'Après-midi', 'Soir', 'Nuit'],
}

for col, categories in ordinal_mappings.items():
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=categories, ordered=True).codes
        print(f" Encodage ordinal : {col}")

# --- One-Hot encoding ---
onehot_cols = [
    'CustomerType', 'FavoriteSeason', 'Region',
    'WeekendPreference', 'ProductDiversity', 'Gender',
    'AccountStatus', 'RFMSegment'
]
onehot_cols = [c for c in onehot_cols if c in df.columns]
df = pd.get_dummies(df, columns=onehot_cols, drop_first=False)
print(f" One-Hot encoding appliqué sur : {onehot_cols}")

# --- Target encoding pour Country ---
if 'Country' in df.columns:
    country_freq = df['Country'].value_counts(normalize=True)
    df['Country'] = df['Country'].map(country_freq)
    print(" Target encoding : Country")

# --- Supprimer CustomerID ---
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)
    print(" Supprimé : CustomerID")

# ============================================================
# 6. SAUVEGARDE DONNÉES NETTOYÉES
# ============================================================
df.to_csv('../data/processed/data_cleaned.csv', index=False)
print("\n Données nettoyées sauvegardées → data/processed/data_cleaned.csv")
print(f"   Shape finale : {df.shape}")

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
# 9. SAUVEGARDE TRAIN/TEST + SCALER
# ============================================================
X_train_scaled.to_csv('../data/train_test/X_train.csv', index=False)
X_test_scaled.to_csv('../data/train_test/X_test.csv',   index=False)
y_train.to_csv('../data/train_test/y_train.csv',         index=False)
y_test.to_csv('../data/train_test/y_test.csv',           index=False)
joblib.dump(scaler, '../models/scaler.joblib')

print("\n Train/Test sauvegardés → data/train_test/")
print(f"   X_train: {X_train_scaled.shape} | X_test: {X_test_scaled.shape}")
print(f"   Distribution Churn train:\n{y_train.value_counts()}")