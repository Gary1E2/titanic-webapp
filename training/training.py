# training.py
import os
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

SHARED_DIR = os.getenv("SHARED_DIR", "/usr/src/app/shared_volume")

def _class_eval(model, X_train, y_train, X_test, y_test, out_png="confusion_matrices.png"):
    plt.figure(figsize=(12, 6))
    data_list = [(X_test, y_test, "Test"), (X_train, y_train, "Train")]

    for i, (X, y, split) in enumerate(data_list):
        pred = model.predict(X)
        cm = confusion_matrix(y, pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\n{split}\n------------------------")
        print(f"Accuracy:    {accuracy_score(y, pred):.4f}")
        print(f"Precision:   {precision_score(y, pred):.4f}")
        print(f"Recall:      {recall_score(y, pred):.4f}")
        print(f"F1 Score:    {f1_score(y, pred):.4f}")
        print(f"Specificity: {tn / (tn + fp):.4f}")

        plt.subplot(1, 2, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='rocket_r',
                    xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel('Predicted Label'); plt.ylabel('True Label')
        plt.title(f'{split} Confusion Matrix')

    out_path = os.path.join(SHARED_DIR, out_png)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_png

def _rocauc_eval(model, X_train, y_train, X_test, y_test, out_png="roc_auc.png"):
    plt.figure(figsize=(12, 6))
    lw = 2
    roc_colours = ['red', 'blue']
    roc_labels = ['Class 0', 'Class 1']

    sets = [(model.predict_proba(X_test), y_test, "Test"),
            (model.predict_proba(X_train), y_train, "Train")]

    for i, (scores_all, y, split) in enumerate(sets):
        plt.subplot(1, 2, i + 1)
        for j in range(2):
            scores = scores_all[:, j]
            fpr, tpr, _ = roc_curve(y == j, scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=lw, color=roc_colours[j],
                     label=f'{roc_labels[j]} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'{split} ROC AUC'); plt.legend(loc="lower right")

    out_path = os.path.join(SHARED_DIR, out_png)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_png

def train_model(
    train_clean_filename="train_clean.csv",
    model_filename="model.pkl"
):
    """Train a model from cleaned train CSV and save model.pkl to SHARED_DIR."""
    train_path = os.path.join(SHARED_DIR, train_clean_filename)
    df = pd.read_csv(train_path)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    param_dict = {
        'max_depth': [3],
        'min_samples_split': [30],
        'min_samples_leaf': [5],
        'max_features': [2, 3, 4],
        'learning_rate': [0.03, 0.05],
        'n_estimators': [120],
        'subsample': [0.8],
        'ccp_alpha': [0.04, 0.03]
    }
    search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_distributions=param_dict,
        n_iter=25, n_jobs=-1, verbose=2, cv=3, random_state=42, scoring='recall'
    )
    search.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(SHARED_DIR, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump(search, f)

    cm_png  = _class_eval(search, X_train, y_train, X_test, y_test, out_png="confusion_matrices.png")
    roc_png = _rocauc_eval(search, X_train, y_train, X_test, y_test, out_png="roc_auc.png")

    metrics = {
        "best_params": search.best_params_,
        "test_score_recall": float(search.score(X_test, y_test)),
        "confusion_png": cm_png,
        "roc_png": roc_png,
        "model_file": model_filename
    }
    return metrics
