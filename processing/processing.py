import os
import numpy as np
import pandas as pd

SHARED_DIR = os.getenv("SHARED_DIR", "/usr/src/app/shared_volume")

def _bin_age_fare(df: pd.DataFrame) -> pd.DataFrame:
    # Age
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    df.loc[df['Age'] <= 16, 'AgeBin'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'AgeBin'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'AgeBin'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'AgeBin'] = 3
    df.loc[df['Age'] > 64, 'AgeBin'] = 4
    df['AgeBin'] = df['AgeBin'].astype(int)

    # Fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    q1 = float(df['Fare'].quantile(0.25))
    q2 = float(df['Fare'].quantile(0.50))
    q3 = float(df['Fare'].quantile(0.75))
    q1_r, q2_r, q3_r = round(q1, 3), round(q2, 3), round(q3, 3)
    df.loc[df['Fare'] <= q1_r, 'FareBin'] = 0
    df.loc[(df['Fare'] > q1_r) & (df['Fare'] <= q2_r), 'FareBin'] = 1
    df.loc[(df['Fare'] > q2_r) & (df['Fare'] <= q3_r), 'FareBin'] = 2
    df.loc[df['Fare'] > q3_r, 'FareBin'] = 3
    df['FareBin'] = df['FareBin'].astype(int)

    # Categorical
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df

def preprocess_pair(train_in="train.csv", test_in="test.csv",
                    train_out="train_clean.csv", test_out="test_clean.csv"):
    train_path = os.path.join(SHARED_DIR, train_in)
    test_path  = os.path.join(SHARED_DIR, test_in)
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_df = _bin_age_fare(train_df)
    test_df  = _bin_age_fare(test_df)

    features_drop = ['Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'PassengerId']
    train_clean = train_df.drop(features_drop, axis=1, errors='ignore')
    test_clean  = test_df.drop(features_drop, axis=1, errors='ignore')

    train_clean.to_csv(os.path.join(SHARED_DIR, train_out), index=False)
    test_clean.to_csv(os.path.join(SHARED_DIR, test_out), index=False)
    return train_out, test_out

def preprocess_single(input_filename: str, output_filename: str = "cleaned.csv"):
    """Clean an arbitrary uploaded file and DROP target if present."""
    inp = os.path.join(SHARED_DIR, input_filename)
    out = os.path.join(SHARED_DIR, output_filename)
    df = pd.read_csv(inp)

    df = _bin_age_fare(df)

    features_drop = ['Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'PassengerId']
    df = df.drop(columns=[c for c in features_drop if c in df.columns], errors='ignore')

    # Drop target if present (users may upload train.csv)
    if 'Survived' in df.columns:
        df = df.drop(columns=['Survived'])

    df.to_csv(out, index=False)
    return output_filename