import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# Chemins absolus pour éviter les erreurs selon d'où on lance
# ============================================================
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, '..', 'models')

# ============================================================
# Charger tous les artefacts
# ============================================================
scaler       = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
classifier   = joblib.load(os.path.join(MODEL_DIR, 'best_classifier.joblib'))
pca          = joblib.load(os.path.join(MODEL_DIR, 'pca.joblib'))
kmeans       = joblib.load(os.path.join(MODEL_DIR, 'kmeans.joblib'))
feat_names   = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))  # 68 colonnes après drop
dropped_cols = joblib.load(os.path.join(MODEL_DIR, 'dropped_cols.joblib'))

CLUSTER_LABELS = {
    0: 'Champions',
    1: 'Fidèles',
    2: 'Potentiels',
    3: 'Dormants'
}

def predict_client(client_data: dict) -> dict:
    """
    Prédit le churn et le segment d'un client.
    Paramètres : client_data (dict) avec les features du client
    Retourne   : dict avec churn_prediction, churn_probability, segment
    """
    # Créer DataFrame
    df = pd.DataFrame([client_data])

    # Ajouter les colonnes manquantes à 0
    for col in feat_names:
        if col not in df.columns:
            df[col] = 0

    # Garder uniquement les 68 colonnes attendues (après suppression multicolinéarité)
    df = df[feat_names].fillna(0)

    # Normalisation avec le scaler entraîné sur X_train réduit
    df_scaled = scaler.transform(df)

    # Prédiction Churn
    churn_pred = classifier.predict(df_scaled)[0]
    churn_prob = classifier.predict_proba(df_scaled)[0][1]

    # Clustering via ACP
    df_pca  = pca.transform(df_scaled)
    cluster = kmeans.predict(df_pca)[0]

    return {
        'churn_prediction': int(churn_pred),
        'churn_probability': round(float(churn_prob) * 100, 2),
        'churn_label':  ' Risque de départ' if churn_pred == 1 else ' Client fidèle',
        'cluster':      int(cluster),
        'segment':      CLUSTER_LABELS.get(int(cluster), 'Inconnu')
    }


if __name__ == '__main__':
    exemple = {
        'Recency':             30,
        'Frequency':            5,
        'MonetaryTotal':      500,
        'Age':                 35,
        'CustomerTenureDays': 365,
        'SatisfactionScore':    4
    }

    result = predict_client(exemple)
    print("\n=== Résultat de prédiction ===")
    for k, v in result.items():
        print(f"  {k:22s} : {v}")