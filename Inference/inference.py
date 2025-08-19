# inference.py
import os
import pickle
import pandas as pd

SHARED_DIR = os.getenv("SHARED_DIR", "/usr/src/app/shared_volume")

def predict_from_clean(
    test_clean_filename="test_clean.csv",
    model_filename="model.pkl",
    output_filename="test_with_preds.csv",
    target_col="Survived"
):
    test_path  = os.path.join(SHARED_DIR, test_clean_filename)
    model_path = os.path.join(SHARED_DIR, model_filename)
    out_path   = os.path.join(SHARED_DIR, output_filename)

    df = pd.read_csv(test_path)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X = df.copy()

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        X = X.reindex(columns=expected, fill_value=0)
    else:
        for col in ["Survived"]:
            if col in X.columns:
                X = X.drop(columns=[col])

    preds = model.predict(X)

    df[target_col] = preds
    df.to_csv(out_path, index=False)
    return output_filename
