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

    test = pd.read_csv(test_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X_test = test.copy()
    test[target_col] = model.predict(X_test)
    test.to_csv(out_path, index=False)
    return output_filename