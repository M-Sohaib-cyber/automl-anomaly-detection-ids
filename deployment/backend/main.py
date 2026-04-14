import os
import json
import joblib
import pandas as pd
import subprocess

from fastapi import FastAPI
from pydantic import BaseModel

print("RUNNING FILE:", __file__)

# ----------------------------
# Paths
# ----------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(BASE, "artifacts")

PREPROCESSOR_PATH = os.path.join(ART, "preprocessor.joblib")
RAW_SCHEMA_PATH = os.path.join(ART, "raw_schema.json")
THRESH_PATH = os.path.join(ART, "threshold.json")

SELECTED_FEATURES_PATH = os.path.join(ART, "selected_features.json")
FINAL_FEATURES_PATH = os.path.join(ART, "final_feature_names.json")

GENMODEL_JAR = os.path.join(ART, "h2o-genmodel.jar")

# Find MOJO zip
mojo_candidates = [f for f in os.listdir(ART) if f.endswith(".zip")]
if not mojo_candidates:
    raise FileNotFoundError("No MOJO .zip found in deployment/artifacts/")
MOJO_PATH = os.path.join(ART, mojo_candidates[0])

# ----------------------------
# Load artifacts
# ----------------------------
preprocessor = joblib.load(PREPROCESSOR_PATH)
raw_schema = json.load(open(RAW_SCHEMA_PATH))
thr = float(json.load(open(THRESH_PATH))["best_f1_threshold"])

selected_features = json.load(open(SELECTED_FEATURES_PATH))
final_feature_names = json.load(open(FINAL_FEATURES_PATH))

# Ensure mandatory categoricals exist
for c in ["proto", "service", "state"]:
    if c not in raw_schema:
        raw_schema.append(c)

print("✅ raw_schema size:", len(raw_schema))
print("✅ final_feature_names size:", len(final_feature_names))
print("✅ selected_features size:", len(selected_features))
print("✅ threshold:", thr)
print("✅ MOJO:", MOJO_PATH)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="AutoML IDS API", version="1.1")


class Record(BaseModel):
    data: dict


@app.get("/")
def root():
    return {"message": "AutoML IDS API running. Visit /docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return {
        "expected_raw_features": raw_schema,
        "threshold": thr,
        "note": "Send raw network features. Missing numeric -> 0, missing categoricals -> 'unknown'."
    }


@app.post("/predict")
def predict(record: Record):
    try:
        df = pd.DataFrame([record.data])

        # 1) Ensure categoricals exist
        for c in ["proto", "service", "state"]:
            if c not in df.columns:
                df[c] = "unknown"
            df[c] = df[c].fillna("unknown").astype(str)

        # 2) Fill missing raw cols
        for col in raw_schema:
            if col not in df.columns:
                if col in ["proto", "service", "state"]:
                    df[col] = "unknown"
                else:
                    df[col] = 0

        # 3) Align raw schema order
        df = df[raw_schema]

        # 4) Preprocess -> numpy matrix
        X = preprocessor.transform(df)

        # 5) Rebuild post-preprocess dataframe with correct column names
        X_df = pd.DataFrame(X, columns=final_feature_names)

        # 6) Keep only MI-selected features used in training
        X_df = X_df[selected_features]

        # 7) Write MOJO input CSV (with headers)
        tmp_in = os.path.join(ART, "_tmp_input.csv")
        tmp_out = os.path.join(ART, "_tmp_output.csv")
        X_df.to_csv(tmp_in, index=False, header=True)

        # 8) Score with MOJO
        cmd = [
            "java",
            "-cp",
            GENMODEL_JAR,
            "hex.genmodel.tools.PredictCsv",
            "--mojo",
            MOJO_PATH,
            "--input",
            tmp_in,
            "--output",
            tmp_out,
            "--decimal",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)

        out = pd.read_csv(tmp_out)

        # Most common binomial output: predict,p0,p1
        if "p1" in out.columns:
            prob_attack = float(out.loc[0, "p1"])
        else:
            # fallback: last numeric column
            num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
            if not num_cols:
                raise RuntimeError(f"MOJO output has no numeric cols: {out.columns.tolist()}")
            prob_attack = float(out.loc[0, num_cols[-1]])

        pred_label = int(prob_attack >= thr)

        return {"prob_attack": prob_attack, "threshold": thr, "pred_label": pred_label}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)