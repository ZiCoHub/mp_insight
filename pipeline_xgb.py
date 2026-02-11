# -*- coding: utf-8 -*-
"""
pipeline_xgb.py
- S1: load/clean views from Excel
- S2: KFold for known/unknown; save OOF; build meta (default threshold), conformal q95 (PI95)
- Train final full-data pipelines and save:
    models/xgb_known_polymer.pkl
    models/xgb_unknown_polymer.pkl
- Save SHAP assets:
    reports/shap_global_known.csv, reports/shap_global_unknown.csv
    models/shap_bg_known.npz, models/shap_bg_unknown.npz
"""

import os, re, json, math, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

# ------------------ Paths (edit only ROOT and RAW_PATH if needed) ------------------
ROOT = Path(r"C:\Users\asus\Desktop\mp-insight")
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "config"
RUNS_DIR = ROOT / "runs_xgb"

RAW_PATH = DATA_DIR / "Complete input.xlsx"     # your Excel path
RAW_SHEET = 0                                   # use first sheet; change if needed

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, CONFIG_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------ Helpers ------------------
def _safe_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def _make_preproc(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[("cat", _safe_ohe(), cat_cols),
                      ("num", "passthrough", num_cols)],
        remainder="drop"
    )

def _metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    # RMSE: be compatible with old sklearn that doesn't have squared=
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


def _normalize_colname(c):
    c = str(c).strip().replace("\u00a0", " ")
    c = re.sub(r"\s+", " ", c)
    return c.replace("Â°C","°C").replace("Celsius","°C")

# ------------------ S1: build views ------------------
TARGET_STD = "logKd"
CATEGORICAL_STD = ["MPs types", "Ageing_Status", "Water Type"]
NUMERIC_STD = [
    "Plastic Density", "SSA", "Particle Size", "LogKow", "E", "S", "A", "B", "V",
    "PH", "Salinity", "Temperature"
]
NAME_MAP = [
    (r"^(logk[_\s-]?d|logk[_\s-]?mp/?w|logk[_\s-]?mpw)$", TARGET_STD),
    (r"^(mps?[\s_/-]?type[s]?)$", "MPs types"),
    (r"^(age(ing)?[\s_-]?(type|status))$", "Ageing_Status"),
    (r"^(water[\s_-]?type)$", "Water Type"),
    (r"^(plastic[\s_-]?density).*", "Plastic Density"),
    (r"^(ssa).*", "SSA"),
    (r"^(particle[\s_-]?size).*", "Particle Size"),
    (r"^(logk[_\s-]?ow|logkow)$", "LogKow"),
    (r"^e$", "E"), (r"^s$", "S"), (r"^a$", "A"), (r"^b$", "B"), (r"^v$", "V"),
    (r"^(ph)$", "PH"),
    (r"^(salinity|salinity[_\s-]?m)$", "Salinity"),
    (r"^(temp.*|temperature.*)", "Temperature"),
]

def _map_columns(cols):
    mapped, used = {}, set()
    low2orig = { _normalize_colname(c).lower(): c for c in cols }
    for patt, std in NAME_MAP:
        rx = re.compile(patt, re.I)
        hit = None
        for low, orig in low2orig.items():
            if orig in used: continue
            if rx.match(low):
                hit = orig; break
        if hit is not None:
            mapped[hit] = std; used.add(hit)
    return mapped

def stage_s1_build_views():
    df_raw = pd.read_excel(RAW_PATH, sheet_name=RAW_SHEET, engine="openpyxl")
    mapped = _map_columns(df_raw.columns)
    df = df_raw.rename(columns=mapped).copy()

    # Basic cleaning
    cat_present = [c for c in CATEGORICAL_STD if c in df.columns]
    for c in cat_present:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # MPs types aliases
    if "MPs types" in df.columns:
        alias = {
            "polystyrene":"PS","ps":"PS",
            "polyethylene":"PE","pe":"PE",
            "polypropylene":"PP","pp":"PP",
            "polyvinyl chloride":"PVC","pvc":"PVC",
            "polyethylene terephthalate":"PET","pet":"PET",
            "polyamide":"PA","pa":"PA","nylon":"PA",
            "polylactic acid":"PLA","pla":"PLA",
            "chlorinated polyethylene":"CPE","cpe":"CPE"
        }
        df["MPs types"] = df["MPs types"].map(lambda x: alias.get(str(x).strip().lower(), x) if pd.notna(x) else x)

    num_present = [c for c in NUMERIC_STD if c in df.columns]
    for c in num_present:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if TARGET_STD not in df.columns:
        raise ValueError(f"Target column '{TARGET_STD}' not found. Found columns: {list(df.columns)}")

    df = df[~df[TARGET_STD].isna()].copy()

    # Build views
    base_cols = [c for c in [TARGET_STD] + cat_present + num_present if c in df.columns]
    df_clean = df[base_cols].copy()

    pollutant_cols = [c for c in ["LogKow","E","S","A","B","V"] if c in df_clean.columns]
    micro_cols     = [c for c in ["MPs types","Ageing_Status","Plastic Density","SSA","Particle Size"] if c in df_clean.columns]
    env_cols       = [c for c in ["PH","Salinity","Temperature","Water Type"] if c in df_clean.columns]

    (DATA_DIR / "views").mkdir(exist_ok=True, parents=True)
    (df_clean).to_csv(DATA_DIR / "dataset_clean.csv", index=False, encoding="utf-8-sig")
    (df_clean).to_csv(DATA_DIR / "view_known.csv", index=False, encoding="utf-8-sig")
    (df_clean.drop(columns=["MPs types"]) if "MPs types" in df_clean.columns else df_clean).to_csv(
        DATA_DIR / "view_unknown.csv", index=False, encoding="utf-8-sig")
    df[[TARGET_STD] + pollutant_cols].to_csv(DATA_DIR / "view_pollutant.csv", index=False, encoding="utf-8-sig")
    df[[TARGET_STD] + micro_cols].to_csv(DATA_DIR / "view_micro.csv", index=False, encoding="utf-8-sig")
    df[[TARGET_STD] + env_cols].to_csv(DATA_DIR / "view_env.csv", index=False, encoding="utf-8-sig")

    print("[S1] Views saved in:", DATA_DIR)

# ------------------ S2: KFold + outputs ------------------
def run_kfold(input_csv: Path, out_root: Path, run_tag: str,
              target: str = "logKd", seed: int = 2025, n_splits: int = 10):

    out_dir = out_root / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if target not in df.columns:
        raise ValueError(f"Missing target '{target}' in {input_csv}")

    feat_cols = [c for c in df.columns if c != target]
    X_raw = df[feat_cols].copy()
    y = df[target].astype(float).values

    cat_candidates = ["MPs types", "Ageing_Status", "Water Type"]
    present_cat = [c for c in cat_candidates if c in X_raw.columns]
    for c in present_cat:
        X_raw[c] = X_raw[c].astype("string")
    num_cols = [c for c in feat_cols if c not in present_cat]
    for c in num_cols:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")

    kf = KFold(n_splits=min(n_splits, max(2, len(X_raw))), shuffle=True, random_state=seed)

    oof = np.zeros(len(X_raw))
    rows = []

    print(f"[KFold] tag={run_tag} | n={len(X_raw)} | folds={kf.get_n_splits()} | cats={present_cat}")
    for fold, (tr, va) in enumerate(kf.split(X_raw), 1):
        Xtr_raw, Xva_raw = X_raw.iloc[tr], X_raw.iloc[va]
        ytr, yva = y[tr], y[va]

        pre = _make_preproc(present_cat, num_cols)
        Xtr = pre.fit_transform(Xtr_raw, ytr)
        Xva = pre.transform(Xva_raw)

        xgb = XGBRegressor(
            n_estimators=2000, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=seed, tree_method="hist"
        )
        xgb.fit(Xtr, ytr)
        pred = xgb.predict(Xva)
        oof[va] = pred

        r2, rm, ma = _metrics(yva, pred)
        rows.append({"fold": fold, "r2": r2, "rmse": rm, "mae": ma})
        print(f"  [Fold {fold:02d}] R2={r2:.4f} | RMSE={rm:.4f} | MAE={ma:.4f}")

    fold_df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "model": "XGBoost",
        "R2_mean": fold_df["r2"].mean(), "R2_std": fold_df["r2"].std(ddof=1),
        "RMSE_mean": fold_df["rmse"].mean(), "RMSE_std": fold_df["rmse"].std(ddof=1),
        "MAE_mean": fold_df["mae"].mean(), "MAE_std": fold_df["mae"].std(ddof=1),
        "n_samples": len(X_raw), "n_folds": kf.get_n_splits()
    }])
    preds_df = pd.DataFrame({"row_id": np.arange(len(X_raw)), "True": y, "Pred": oof})

    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    preds_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    print("[KFold] Finished. Saved per-fold, summary, predictions at:", out_dir)
    return out_dir

# ------------------ Meta + Conformal (PI95) ------------------
def save_model_meta(meta_path: Path,
                    default_threshold: float = 4.0,
                    csolid_presets = (1e-5, 1e-4, 1e-3),
                    default_R: float = 0.8):
    meta = {
        "default_threshold": float(default_threshold),
        "risk_bands": {"low": "<2", "medium": "[2,4)", "high": ">=4"},
        "csolid_presets": list(map(float, csolid_presets)),
        "default_R": float(default_R),
        "notes": "Based on sorbed fraction >=~80% under high MP concentration (Csolid ≈ 1e-3 g/mL)"
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def build_conformal_from_oof(oof_csv_path: Path, out_json_path: Path,
                             q: float = 0.95, true_col: str = "True", pred_col: str = "Pred"):
    oof = pd.read_csv(oof_csv_path)
    if true_col not in oof.columns or pred_col not in oof.columns:
        raise ValueError(f"OOF file must contain '{true_col}' and '{pred_col}': {oof_csv_path}")
    abs_err = (oof[true_col].astype(float) - oof[pred_col].astype(float)).abs().values
    qv = float(np.quantile(abs_err, q))
    out = {"q95": qv, "n": int(len(abs_err))}
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[Conformal] q95={qv:.4f} saved to {out_json_path}")

# ------------------ Train full-data pipelines + SHAP assets ------------------
def train_full_and_save(input_csv: Path, model_out: Path, shap_bg_out: Path, shap_global_csv: Path,
                        target: str = "logKd", bg_max_n: int = 500, seed: int = 42):
    df = pd.read_csv(input_csv)
    if target not in df.columns:
        raise ValueError(f"Missing target '{target}' in {input_csv}")

    y = df.pop(target).astype(float).values
    cat_candidates = ["MPs types", "Ageing_Status", "Water Type"]
    present_cat = [c for c in cat_candidates if c in df.columns]
    for c in present_cat:
        df[c] = df[c].astype("string")
    num_cols = [c for c in df.columns if c not in present_cat]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    pre = _make_preproc(present_cat, num_cols)
    xgb = XGBRegressor(
        n_estimators=1200, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=seed, tree_method="hist"
    )
    pipe = Pipeline([("pre", pre), ("model", xgb)])
    pipe.fit(df, y)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)
    print("[Model] Saved pipeline to", model_out)

    # ---- SHAP assets (global ranking + background) ----
    try:
        import shap
        # Build a small background in preprocessed space (sample rows)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df))
        if len(idx) > bg_max_n:
            idx = rng.choice(idx, size=bg_max_n, replace=False)
        X_bg = pipe.named_steps["pre"].transform(df.iloc[idx])
        # Save background
        np.savez_compressed(shap_bg_out, bg_shape=X_bg.shape, bg=X_bg)
        print("[SHAP] Background saved to", shap_bg_out)

        # Global ranking via mean |shap| on a small subset
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        # Need model input in tree space; use booster input (preprocessed)
        X_sample = pipe.named_steps["pre"].transform(df.iloc[idx])
        shap_values = explainer.shap_values(X_sample)
        # Map feature names
        try:
            feat_names = list(pipe.named_steps["pre"].get_feature_names_out())
        except Exception:
            feat_names = [f"f{i}" for i in range(X_sample.shape[1])]

        mean_abs = np.mean(np.abs(shap_values), axis=0).ravel()
        top = pd.DataFrame({"Feature": feat_names, "MeanAbsSHAP": mean_abs}).sort_values(
            "MeanAbsSHAP", ascending=False)
        shap_global_csv.parent.mkdir(parents=True, exist_ok=True)
        top.to_csv(shap_global_csv, index=False, encoding="utf-8-sig")
        print("[SHAP] Global ranking saved to", shap_global_csv)
    except Exception as e:
        print("[SHAP] Skipped (install shap if needed). Reason:", e)

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all",
                        choices=["s1","s2","all"],
                        help="s1: build views; s2: kfold + meta + conformal + models + SHAP; all: both")
    args = parser.parse_args()

    if args.stage in ("s1","all"):
        stage_s1_build_views()

    if args.stage in ("s2","all"):
        # KFold on known view
        known_csv = DATA_DIR / "view_known.csv"
        unknown_csv = DATA_DIR / "view_unknown.csv"

        out_known = run_kfold(known_csv, RUNS_DIR, "known_kfold", target="logKd")
        out_unknown = run_kfold(unknown_csv, RUNS_DIR, "unknown_kfold", target="logKd")

        # Meta + Conformal (PI95) for known & unknown
        save_model_meta(CONFIG_DIR / "model_meta_known.json", default_threshold=4.0,
                        csolid_presets=(1e-5,1e-4,1e-3), default_R=0.8)
        save_model_meta(CONFIG_DIR / "model_meta_unknown.json", default_threshold=4.0,
                        csolid_presets=(1e-5,1e-4,1e-3), default_R=0.8)

        build_conformal_from_oof(out_known / "predictions.csv", MODELS_DIR / "conformal_known.json", q=0.95)
        build_conformal_from_oof(out_unknown / "predictions.csv", MODELS_DIR / "conformal_unknown.json", q=0.95)

        # Train full pipelines + SHAP assets
        train_full_and_save(known_csv,
                            MODELS_DIR / "xgb_known_polymer.pkl",
                            MODELS_DIR / "shap_bg_known.npz",
                            REPORTS_DIR / "shap_global_known.csv",
                            target="logKd")
        train_full_and_save(unknown_csv,
                            MODELS_DIR / "xgb_unknown_polymer.pkl",
                            MODELS_DIR / "shap_bg_unknown.npz",
                            REPORTS_DIR / "shap_global_unknown.csv",
                            target="logKd")

    print("\nDone.")

if __name__ == "__main__":
    main()
