import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def load_data(path):
    """Charge les données depuis un CSV"""
    df = pd.read_csv(path)
    print(f" Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def missing_report(df):
    """Rapport sur les valeurs manquantes"""
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    report = pd.DataFrame({'Manquants': missing, 'Pourcentage': pct})
    return report[report['Manquants'] > 0].sort_values('Pourcentage', ascending=False)

def fix_support_tickets(df):
    """Remplace les valeurs aberrantes de SupportTicketsCount (999 et -1 → NaN)"""
    df = df.copy()
    if 'SupportTicketsCount' in df.columns:
        df['SupportTicketsCount'] = df['SupportTicketsCount'].replace({999: np.nan, -1: np.nan})
        print(" SupportTicketsCount nettoyé")
    return df

def fix_satisfaction(df):
    """Remplace les valeurs aberrantes de SatisfactionScore (-1, 99, 0 → NaN)"""
    df = df.copy()
    if 'SatisfactionScore' in df.columns:
        df['SatisfactionScore'] = df['SatisfactionScore'].replace({-1: np.nan, 99: np.nan, 0: np.nan})
        print(" SatisfactionScore nettoyé")
    return df

def parse_registration_date(df):
    """Parse RegistrationDate et extrait des features"""
    df = df.copy()
    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
        df['RegYear']    = df['RegistrationDate'].dt.year
        df['RegMonth']   = df['RegistrationDate'].dt.month
        df['RegDay']     = df['RegistrationDate'].dt.day
        df['RegWeekday'] = df['RegistrationDate'].dt.weekday
        df.drop(columns=['RegistrationDate'], inplace=True)
        print(" RegistrationDate parsée et supprimée")
    return df

def parse_last_login_ip(df):
    """Extrait des features depuis LastLoginIP"""
    df = df.copy()
    if 'LastLoginIP' in df.columns:
        df['IP_FirstOctet'] = df['LastLoginIP'].str.split('.').str[0].astype(float)
        df['IsPrivateIP'] = df['LastLoginIP'].apply(
            lambda x: 1 if str(x).startswith(('192.168', '10.', '172.')) else 0
        )
        df.drop(columns=['LastLoginIP'], inplace=True)
        print(" LastLoginIP parsée et supprimée")
    return df

def drop_useless_features(df):
    """Supprime les features inutiles"""
    df = df.copy()
    if 'NewsletterSubscribed' in df.columns:
        df.drop(columns=['NewsletterSubscribed'], inplace=True)
        print(" Supprimé : NewsletterSubscribed")
    return df

def feature_engineering(df):
    """Crée de nouvelles features"""
    df = df.copy()
    df['MonetaryPerDay']  = df['MonetaryTotal'] / (df['Recency'] + 1)
    df['AvgBasketValue']  = df['MonetaryTotal'] / df['Frequency']
    df['TenureRatio']     = df['Recency'] / (df['CustomerTenureDays'] + 1)
    print(" Feature engineering terminé")
    return df

def impute_missing(df, numeric_cols, strategy='median'):
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            if strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
    print(" Imputation terminée")
    return df