import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve,
                              silhouette_score, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. CHARGEMENT DES DONNÉES TRAIN/TEST
# ============================================================
X_train = pd.read_csv('../data/train_test/X_train.csv')
X_test  = pd.read_csv('../data/train_test/X_test.csv')
y_train = pd.read_csv('../data/train_test/y_train.csv').squeeze()
y_test  = pd.read_csv('../data/train_test/y_test.csv').squeeze()

print(f"Données chargées")
print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"   Distribution Churn train: {y_train.value_counts().to_dict()}")

# ============================================================
# 2. MULTICOLINÉARITÉ — seuil 0.8 (exigé par la prof)
# ============================================================
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]

print(f"\n Features supprimées (corrélation > 0.8) : {to_drop}")
X_train = X_train.drop(columns=to_drop)
X_test  = X_test.drop(columns=to_drop)
print(f"   Shape après suppression multicolinéarité : {X_train.shape}")

# Sauvegarder la liste des colonnes finales
joblib.dump(X_train.columns.tolist(), '../models/feature_names.joblib')
joblib.dump(to_drop, '../models/dropped_cols.joblib')

# ============================================================
# 3. DÉSÉQUILIBRE DES CLASSES — SMOTE (exigé par la prof)
# ============================================================
print(f"\n Avant SMOTE : {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f" Après SMOTE  : {pd.Series(y_train_bal).value_counts().to_dict()}")

# ============================================================
# 4. ACP — RÉDUCTION DE DIMENSION (fit sur train uniquement)
# ============================================================
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_bal)
X_test_pca  = pca.transform(X_test)

n_comp = X_train_pca.shape[1]
print(f"\n ACP : {X_train.shape[1]} features → {n_comp} composantes (95% variance)")

# Graphique variance expliquée
plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='steelblue')
plt.axhline(y=0.95, color='red', linestyle='--', label='Seuil 95%')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance expliquée cumulée')
plt.title('ACP — Variance expliquée cumulée')
plt.legend()
plt.tight_layout()
plt.savefig('../reports/pca_variance.png')
plt.show()

joblib.dump(pca, '../models/pca.joblib')

# ============================================================
# 5. CLUSTERING K-MEANS (sur données ACP)
# ============================================================
inertias    = []
silhouettes = []
K_range     = range(2, 9)

for k in K_range:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_train_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_train_pca, labels))

# Graphique coude + silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, marker='o', color='coral')
axes[0].set_title('Méthode du Coude — K-Means')
axes[0].set_xlabel('Nombre de clusters K')
axes[0].set_ylabel('Inertie')

axes[1].plot(K_range, silhouettes, marker='o', color='steelblue')
axes[1].set_title('Score Silhouette')
axes[1].set_xlabel('Nombre de clusters K')
axes[1].set_ylabel('Score Silhouette')

plt.tight_layout()
plt.savefig('../reports/kmeans_elbow.png')
plt.show()

# K=4 correspond aux 4 segments RFM du projet (Champions, Fidèles, Potentiels, Dormants)
K_OPTIMAL = 4
kmeans        = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
clusters_train = kmeans.fit_predict(X_train_pca)
clusters_test  = kmeans.predict(X_test_pca)

print(f"\n K-Means : {K_OPTIMAL} clusters")
print(f"   Distribution : {pd.Series(clusters_train).value_counts().to_dict()}")

# Visualisation 2D
pca_2d = PCA(n_components=2, random_state=42)
X_2d   = pca_2d.fit_transform(X_train_pca)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                      c=clusters_train, cmap='tab10', alpha=0.5, s=15)
plt.colorbar(scatter, label='Cluster')
plt.title('Visualisation K-Means en 2D (via ACP)')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.tight_layout()
plt.savefig('../reports/kmeans_clusters_2d.png')
plt.show()

joblib.dump(kmeans, '../models/kmeans.joblib')

# ============================================================
# 6. CLASSIFICATION — PRÉDICTION DU CHURN
# ============================================================
# class_weight='balanced' en plus de SMOTE pour robustesse
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5)
}

results = {}

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        'model':     model,
        'auc':       auc,
        'f1':        report['weighted avg']['f1-score'],
        'precision': report['weighted avg']['precision'],
        'recall':    report['weighted avg']['recall']
    }

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred))
    print(f"  ROC-AUC : {auc:.4f}")

# Tableau comparatif
results_df = pd.DataFrame({
    name: {
        'AUC':       v['auc'],
        'F1':        v['f1'],
        'Precision': v['precision'],
        'Recall':    v['recall']
    }
    for name, v in results.items()
}).T.sort_values('AUC', ascending=False)

print("\n Comparaison des modèles :")
print(results_df.round(4))
results_df.to_csv('../reports/models_comparison.csv')

# ============================================================
# 7. OPTIMISATION HYPERPARAMÈTRES — GridSearchCV
# ============================================================
print("\n GridSearchCV sur Random Forest...")
param_grid = {
    'n_estimators':      [100, 200],
    'max_depth':         [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_bal, y_train_bal)

best_model = grid_search.best_estimator_
print(f" Meilleurs hyperparamètres : {grid_search.best_params_}")
print(f"   Meilleur AUC (CV) : {grid_search.best_score_:.4f}")

# ============================================================
# 8. ÉVALUATION FINALE
# ============================================================
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fidèle (0)', 'Churn (1)'],
            yticklabels=['Fidèle (0)', 'Churn (1)'])
plt.title('Matrice de Confusion — Meilleur Modèle')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.tight_layout()
plt.savefig('../reports/confusion_matrix.png')
plt.show()

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
auc_final   = roc_auc_score(y_test, y_prob_best)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC = {auc_final:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.50)')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC — Meilleur Modèle')
plt.legend()
plt.tight_layout()
plt.savefig('../reports/roc_curve.png')
plt.show()

# Feature Importance
feat_imp = pd.Series(best_model.feature_importances_, index=X_train.columns)
top20 = feat_imp.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 7))
top20.sort_values().plot(kind='barh', color='steelblue')
plt.title('Top 20 Features les plus importantes')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('../reports/feature_importance.png')
plt.show()

print(f"\n Évaluation finale — AUC : {auc_final:.4f}")
print(classification_report(y_test, y_pred_best))

# ============================================================
# 9. RÉGRESSION — PRÉDICTION MonetaryTotal
# ============================================================
print("\n Régression — Prédiction MonetaryTotal")

# On utilise X_train/X_test déjà préparés (après suppression multicolinéarité)
# La target régression est dans X_train si elle n'a pas été droppée
# On recharge processed pour extraire y_reg proprement
df_proc   = pd.read_csv('../data/processed/data_cleaned.csv')
df_proc   = df_proc.drop(columns=to_drop, errors='ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SC

target_reg = 'MonetaryTotal'
X_reg = df_proc.drop(columns=[target_reg, 'Churn', 'CustomerID'], errors='ignore')
y_reg = df_proc[target_reg]

X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg   = SC()
X_r_train_sc = scaler_reg.fit_transform(X_r_train)
X_r_test_sc  = scaler_reg.transform(X_r_test)

reg_models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Ridge (alpha=10)':  Ridge(alpha=10.0),
    'Random Forest Reg': RandomForestRegressor(n_estimators=100, random_state=42)
}

reg_results = {}
for name, model in reg_models.items():
    model.fit(X_r_train_sc, y_r_train)
    y_pred_r = model.predict(X_r_test_sc)
    rmse = np.sqrt(mean_squared_error(y_r_test, y_pred_r))
    r2   = r2_score(y_r_test, y_pred_r)
    reg_results[name] = {'RMSE': round(rmse, 2), 'R2': round(r2, 4)}
    print(f"  {name:25s} → RMSE: {rmse:.2f} | R²: {r2:.4f}")

# Sauvegarder le meilleur modèle de régression (Random Forest)
best_reg = RandomForestRegressor(n_estimators=100, random_state=42)
best_reg.fit(X_r_train_sc, y_r_train)
joblib.dump(best_reg,   '../models/regression_model.joblib')
joblib.dump(scaler_reg, '../models/scaler_regression.joblib')
joblib.dump(X_reg.columns.tolist(), '../models/feature_names_reg.joblib')

# ============================================================
# 10. SAUVEGARDE DU MEILLEUR CLASSIFIEUR
# ============================================================
joblib.dump(best_model, '../models/best_classifier.joblib')

print("\n Tous les modèles sauvegardés dans models/")
print("   best_classifier.joblib")
print("   kmeans.joblib")
print("   pca.joblib")
print("   regression_model.joblib")
print("   scaler_regression.joblib")
print("   feature_names.joblib")
print("   dropped_cols.joblib")