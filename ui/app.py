import os, time, random
import pandas as pd, requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename

# ---- Config
SHARED_DIR     = os.getenv("SHARED_DIR", "/usr/src/app/shared_volume")
PROCESSING_URL = os.getenv("PROCESSING_URL", "http://localhost:8000")
TRAINING_URL   = os.getenv("TRAINING_URL",   "http://localhost:8001")
INFERENCE_URL  = os.getenv("INFERENCE_URL",  "http://localhost:8002")

for d in ("results", "static", SHARED_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.secret_key = "dev-secret"

FEATURES = ["Sex", "Pclass", "Embarked", "Age", "Fare"]
TARGET   = "Survived"

# ---- Utils
ts = lambda: str(int(time.time() * 1000))

def read_csv(path):
    try: return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError: return pd.read_csv(path, encoding="latin-1", low_memory=False)

def norm(s): return s.lower().replace(" ", "").replace("_", "")

def suggest_map(cols):
    m, n = {k: None for k in (FEATURES + [TARGET])}, {norm(c): c for c in cols}
    def f(*cands):
        for c in cands:
            v = n.get(norm(c))
            if v: return v
    m["Sex"] = f("sex","gender","male_female")
    m["Pclass"] = f("pclass","passengerclass","class")
    m["Embarked"] = f("embarked","port","embark","embarkation")
    m["Age"] = f("age","years")
    m["Fare"] = f("fare","ticketprice","price")
    m[TARGET] = f("survived","label","target","y")
    return m

def apply_map(df, m):
    ren = {src: exp for exp, src in m.items() if src and src in df.columns and exp != src}
    df = df.rename(columns=ren)
    return df.loc[:, ~df.columns.duplicated(keep="first")]

def add_check(v, name, ok, details):
    v["checks"].append({"name": name, "ok": bool(ok), "details": str(details)})
    if not ok: v["status"] = "fail"

def validate_df(df, target_mapped):
    v = {"status":"pass","checks":[],"summary":{}}
    try:
        miss = [c for c in FEATURES if c not in df.columns]
        add_check(v, "Required features", len(miss)==0, "OK" if not miss else f"Missing: {miss}")
    except Exception as e: add_check(v, "Required features", False, e)

    for col in ("Age","Fare"):
        try:
            ok = col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()
            add_check(v, f"{col} numeric", ok, "OK" if ok else "Not numeric/coercible or missing")
        except Exception as e: add_check(v, f"{col} numeric", False, e)

    for col in ("Sex","Embarked","Pclass"):
        try:
            if col in df.columns:
                obj = df[col]
                miss = int(obj.isna().sum().sum() if isinstance(obj, pd.DataFrame) else obj.isna().sum())
                total = df.shape[0]
                add_check(v, f"{col} non-null", miss < total, f"Missing {miss}/{total}")
            else:
                add_check(v, f"{col} non-null", False, "Column not found")
        except Exception as e: add_check(v, f"{col} non-null", False, e)

    try:
        add_check(v, f"Target ({TARGET})", True,
                  "Present (dropped before inference)" if TARGET in df.columns or target_mapped else "Not provided")
    except Exception as e: add_check(v, f"Target ({TARGET})", False, e)

    try:
        v["summary"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_counts": {c: int((df[c].isna().sum().sum() if isinstance(df[c], pd.DataFrame) else df[c].isna().sum()))
                               for c in FEATURES if c in df.columns}
        }
    except Exception as e: add_check(v, "Summary", False, e)
    return v

def pie(series, title, fname):
    plt.figure(figsize=(5,5))
    plt.pie(series, labels=series.index, autopct="%1.1f%%", startangle=140)
    plt.title(title); plt.tight_layout()
    out = os.path.join("static", fname); plt.savefig(out); plt.close(); return out

def grouped(df, group_col, pred_col, fname):
    counts = df.groupby([group_col, pred_col]).size().unstack(fill_value=0)
    plt.figure(figsize=(7,4)); counts.plot(kind="bar")
    plt.title(f"{pred_col} by {group_col}"); plt.xlabel(group_col); plt.ylabel("Count"); plt.tight_layout()
    out = os.path.join("static", fname); plt.savefig(out); plt.close(); return out

def pick_group(df):
    for c in ("Sex","Pclass","Embarked","AgeBin","FareBin"):
        if c in df.columns: return c

def copy_artifact(name):
    src = os.path.join(SHARED_DIR, name)
    if os.path.exists(src):
        dst = os.path.join("static", f"{os.path.splitext(name)[0]}_{ts()}.png")
        with open(src,"rb") as rf, open(dst,"wb") as wf: wf.write(rf.read())
        return dst

def summarize(df):
    return {"rows": len(df), "columns": len(df.columns), "missing_values": int(df.isnull().sum().sum())}

# ---- Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Please select a CSV"); return redirect(url_for("index"))
    filename = secure_filename(f.filename)
    path = os.path.join(SHARED_DIR, filename); f.save(path)
    df = read_csv(path)
    return render_template("map.html",
        filename=filename,
        columns=list(df.columns),
        expected=(FEATURES + [TARGET]),
        target_name=TARGET,
        suggested=suggest_map(df.columns)
    )

@app.route("/apply-mapping", methods=["POST"])
def apply_mapping():
    filename = request.form.get("filename")
    if not filename:
        flash("Missing filename"); return redirect(url_for("index"))
    src = os.path.join(SHARED_DIR, filename)
    df_raw = read_csv(src)

    mapping = {k: (request.form.get(k) or None) for k in (FEATURES + [TARGET])}
    target_mapped = mapping.get(TARGET) is not None

    chosen = [v for v in mapping.values() if v]
    dupes = sorted({s for s in chosen if chosen.count(s) > 1})
    if dupes:
        flash(f"Each CSV column can be mapped once. Duplicates: {', '.join(dupes)}")
        return render_template("map.html",
            filename=filename, columns=list(df_raw.columns),
            expected=(FEATURES + [TARGET]), target_name=TARGET, suggested=mapping
        )

    df_map = apply_map(df_raw, mapping)
    validation = validate_df(df_map, target_mapped)

    mapped_name = f"mapped_{ts()}_{filename}"
    mapped_path = os.path.join(SHARED_DIR, mapped_name)
    df_map.to_csv(mapped_path, index=False)

    cleaned_name = f"cleaned_{ts()}.csv"
    cleaned_path = os.path.join(SHARED_DIR, cleaned_name)
    processed_ok = False
    try:
        r = requests.post(f"{PROCESSING_URL}/process/single",
                          json={"input_filename": mapped_name, "output_filename": cleaned_name},
                          timeout=300); r.raise_for_status()
        processed_ok = True
    except requests.HTTPError as e:
        add_check(validation, "Processing service", False, f"HTTP {e.response.status_code}: {e.response.text}")
    except Exception as e:
        add_check(validation, "Processing service", False, f"{type(e).__name__}: {e}")

    if not os.path.exists(os.path.join(SHARED_DIR, "model.pkl")):
        try:
            if not os.path.exists(os.path.join(SHARED_DIR, "train_clean.csv")):
                r1 = requests.post(f"{PROCESSING_URL}/process/pair",
                                   json={"train_in":"train.csv","test_in":"test.csv",
                                         "train_out":"train_clean.csv","test_out":"test_clean.csv"},
                                   timeout=600); r1.raise_for_status()
            rt = requests.post(f"{TRAINING_URL}/train",
                               json={"train_clean_filename":"train_clean.csv","model_filename":"model.pkl"},
                               timeout=1800); rt.raise_for_status()
        except Exception: pass

    preds_file = f"preds_{ts()}.csv"
    preds_path = os.path.join(SHARED_DIR, preds_file)
    used_live = False
    df_fb = read_csv(cleaned_path) if (processed_ok and os.path.exists(cleaned_path)) else df_map.copy()

    try:
        ri = requests.post(f"{INFERENCE_URL}/predict",
                           json={"test_clean_filename": cleaned_name if processed_ok else mapped_name,
                                 "model_filename": "model.pkl",
                                 "output_filename": preds_file,
                                 "target_col": "Prediction"},
                           timeout=600); ri.raise_for_status()
        df_pred = read_csv(preds_path); used_live = True
    except requests.HTTPError as e:
        add_check(validation, "Inference service", False, f"HTTP {e.response.status_code}: {e.response.text}")
        df_pred = df_fb.copy(); df_pred["Prediction"] = [random.choice(["Positive","Negative"]) for _ in range(len(df_fb))]
        df_pred.to_csv(preds_path, index=False)
    except Exception as e:
        add_check(validation, "Inference service", False, f"{type(e).__name__}: {e}")
        df_pred = df_fb.copy(); df_pred["Prediction"] = [random.choice(["Positive","Negative"]) for _ in range(len(df_fb))]
        df_pred.to_csv(preds_path, index=False)

    summary = summarize(df_pred)
    charts = []
    if "Prediction" in df_pred.columns:
        counts = df_pred["Prediction"].value_counts()
        charts.append(pie(counts, "Prediction Distribution", f"pred_pie_{ts()}.png"))
        g = pick_group(df_pred)
        if g: charts.append(grouped(df_pred, g, "Prediction", f"grouped_{g}_{ts()}.png"))
    roc = copy_artifact("roc_auc.png")
    if roc: charts.append(roc)

    dl_preds = f"results_{filename}"
    df_pred.to_csv(os.path.join("results", dl_preds), index=False)

    dl_cleaned = None
    if processed_ok and os.path.exists(cleaned_path):
        dl_cleaned = f"{os.path.splitext(cleaned_name)[0]}_{filename}"
        read_csv(cleaned_path).to_csv(os.path.join("results", dl_cleaned), index=False)

    preview = df_pred.head(100)
    return render_template("display.html",
        filename=filename,
        used_live_model=used_live,
        tables=[preview.to_html(classes="table table-bordered table-hover text-center", index=False)],
        summary=summary,
        validation=validation,
        chart_urls=[url_for("static", filename=os.path.basename(p), v=ts()) for p in charts],
        download_url=url_for("download", filename=dl_preds),
        cleaned_download_url=(url_for("download", filename=dl_cleaned) if dl_cleaned else None)
    )

@app.route("/run-titanic")
def run_titanic():
    try:
        r1 = requests.post(f"{PROCESSING_URL}/process/pair",
                           json={"train_in":"train.csv","test_in":"test.csv",
                                 "train_out":"train_clean.csv","test_out":"test_clean.csv"},
                           timeout=600); r1.raise_for_status()
        r2 = requests.post(f"{TRAINING_URL}/train",
                           json={"train_clean_filename":"train_clean.csv","model_filename":"model.pkl"},
                           timeout=1800); r2.raise_for_status()
        out_csv = f"test_with_preds_{ts()}.csv"
        r3 = requests.post(f"{INFERENCE_URL}/predict",
                           json={"test_clean_filename":"test_clean.csv",
                                 "model_filename":"model.pkl",
                                 "output_filename": out_csv,
                                 "target_col":"Survived"},
                           timeout=600); r3.raise_for_status()
        df = read_csv(os.path.join(SHARED_DIR, out_csv))
        used_live = True
    except Exception as e:
        flash(f"Pipeline failed: {e}")
        return redirect(url_for("index"))

    summary = summarize(df)
    charts = []
    if "Survived" in df.columns:
        charts.append(pie(df["Survived"].value_counts(), "Survival Distribution", f"surv_pie_{ts()}.png"))
        g = pick_group(df)
        if g: charts.append(grouped(df, g, "Survived", f"grouped_{g}_{ts()}.png"))
    roc = copy_artifact("roc_auc.png")
    if roc: charts.append(roc)

    dl_preds = "titanic_predictions.csv"
    df.to_csv(os.path.join("results", dl_preds), index=False)

    preview = df.head(100)
    return render_template("display.html",
        filename="test.csv",
        used_live_model=used_live,
        tables=[preview.to_html(classes="table table-bordered table-hover text-center", index=False)],
        summary=summary, validation=None,
        chart_urls=[url_for("static", filename=os.path.basename(p), v=ts()) for p in charts],
        download_url=url_for("download", filename=dl_preds),
        cleaned_download_url=None
    )

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join("results", filename), as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
