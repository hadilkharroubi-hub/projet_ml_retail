import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def load_data(path):
    """Charge les données depuis un CSV"""
    df = pd.read_csv(path)
    print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def missing_report(df):
    """Rapport sur les valeurs manquantes"""
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    report = pd.DataFrame({'Manquants': missing, 'Pourcentage': pct})
    return report[report['Manquants'] > 0].sort_values('Pourcentage', ascending=False)

def fix_support_tickets(df):
    """Remplace les valeurs aberrantes de SupportTickets (999 et -1 → NaN)"""
    df = df.copy()
    df['SupportTickets'] = df['SupportTickets'].replace({999: np.nan, -1: np.nan})
    return df

def fix_satisfaction(df):
    """Remplace les valeurs aberrantes de Satisfaction (-1, 99 → NaN)"""
    df = df.copy()
    df['Satisfaction'] = df['Satisfaction'].replace({-1: np.nan, 99: np.nan, 0: np.nan})
    return df

def parse_registration_date(df):
    """Parse la colonne RegistrationDate en datetime et extrait des features"""
    df = df.copy()
    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
    df['RegYear']    = df['RegistrationDate'].dt.year
    df['RegMonth']   = df['RegistrationDate'].dt.month
    df['RegDay']     = df['RegistrationDate'].dt.day
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday
    df.drop(columns=['RegistrationDate'], inplace=True)
    return df

def parse_last_login_ip(df):
    """Extrait des features depuis LastLoginIP"""
    df = df.copy()
    df['IP_FirstOctet'] = df['LastLoginIP'].str.split('.').str[0].astype(float)
    df['IsPrivateIP'] = df['LastLoginIP'].apply(
        lambda x: 1 if str(x).startswith(('192.168', '10.', '172.')) else 0
    )
    df.drop(columns=['LastLoginIP'], inplace=True)
    return df

def drop_useless_features(df):
    """Supprime les features inutiles (variance nulle, etc.)"""
    df = df.copy()
    if 'NewsletterSubscribed' in df.columns:
        df.drop(columns=['NewsletterSubscribed'], inplace=True)
        print("✅ Supprimé : NewsletterSubscribed")
    return df

def feature_engineering(df):
    """Crée de nouvelles features"""
    df = df.copy()
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']
    df['TenureRatio']    = df['Recency'] / (df['CustomerTenure'] + 1)
    return df

def impute_missing(df, numeric_cols, strategy='median'):
    """Impute les valeurs manquantes pour colonnes numériques"""
    df = df.copy()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
    return df