"""
src/model_training.py  —  FraudSentinel 8-Model Training Arena
===============================================================
Trains: XGBoost, LightGBM, CatBoost, Random Forest,
        Logistic Regression, SVM, MLP, Isolation Forest
Saves leaderboard, all model joblibs, and comparison metrics.
"""
import os, sys, warnings, time
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, IsolationForest
from sklearn.svm             import SVC
from sklearn.neural_network  import MLPClassifier
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier
from catboost                import CatBoostClassifier
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score,
    matthews_corrcoef, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)
from src.data_processing import load_and_preprocess

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Evaluation helper ──────────────────────────────────────────────────────────
def _eval(y_test, y_pred, y_prob, feat_names, model_obj, train_time):
    pc, rc, _ = precision_recall_curve(y_test, y_prob)
    pr_auc  = auc(rc, pc)
    roc_auc = roc_auc_score(y_test, y_prob)
    mcc     = matthews_corrcoef(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)
    tn,fp,fn,tp = cm.ravel()
    sens    = tp / (tp + fn) if (tp + fn) else 0
    spec    = tn / (tn + fp) if (tn + fp) else 0
    g_mean  = float(np.sqrt(sens * spec))

    # Feature importance (if available)
    fi = None
    if hasattr(model_obj, "feature_importances_"):
        fi = dict(zip(feat_names, model_obj.feature_importances_.tolist()))
    elif hasattr(model_obj, "coef_"):
        fi = dict(zip(feat_names, np.abs(model_obj.coef_[0]).tolist()))

    return {
        "pr_auc":            round(pr_auc, 4),
        "roc_auc":           round(roc_auc, 4),
        "f1_score":          round(f1, 4),
        "precision":         round(prec, 4),
        "recall":            round(rec, 4),
        "sensitivity":       round(sens, 4),
        "specificity":       round(spec, 4),
        "g_mean":            round(g_mean, 4),
        "mcc":               round(mcc, 4),
        "confusion_matrix":  cm.tolist(),
        "precision_curve":   pc.tolist(),
        "recall_curve":      rc.tolist(),
        "feature_names":     feat_names,
        "feature_importance":fi,
        "train_time_s":      round(train_time, 2),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ── Inference timing helper ────────────────────────────────────────────────────
def _infer_latency(model, X_sample):
    t0 = time.perf_counter()
    for _ in range(100):
        model.predict_proba(X_sample[:1])
    return round((time.perf_counter() - t0) / 100 * 1000, 3)  # ms


# ── Main training function ─────────────────────────────────────────────────────
def train_model(random_state=42):
    X_train_res, y_train_res, X_test, y_test, scaler, feat_names = \
        load_and_preprocess(random_state=random_state)

    X_test_df = pd.DataFrame(X_test, columns=feat_names)
    neg = int((y_train_res == 0).sum())
    pos = int((y_train_res == 1).sum())

    # ── MODEL DEFINITIONS ──────────────────────────────────────────────────────
    models_cfg = [

        ("XGBoost", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80,
            scale_pos_weight=neg/pos, use_label_encoder=False,
            eval_metric="aucpr", random_state=random_state,
            n_jobs=-1, tree_method="hist", verbosity=0,
        )),

        ("LightGBM", LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg/pos, random_state=random_state,
            n_jobs=-1, verbose=-1,
        )),

        ("CatBoost", CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=random_state, verbose=0,
        )),

        ("Random Forest", RandomForestClassifier(
            n_estimators=300, max_depth=12,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )),

        ("Logistic Regression", LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )),

        ("SVM", SVC(
            kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", probability=True,
            random_state=random_state,
        )),

        ("MLP Neural Network", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu", max_iter=300,
            random_state=random_state, early_stopping=True,
            validation_fraction=0.1,
        )),
    ]

    comparison  = {}
    leaderboard = []
    primary_xgb = None
    primary_metrics = None

    for name, model in models_cfg:
        print(f"\n[training] {name} ...")
        t0 = time.perf_counter()

        if name == "XGBoost":
            model.fit(X_train_res, y_train_res,
                      eval_set=[(X_test, y_test)], verbose=50)
        elif name == "CatBoost":
            model.fit(X_train_res, y_train_res,
                      eval_set=(X_test, y_test))
        elif name == "LightGBM":
            model.fit(X_train_res, y_train_res,
                      eval_set=[(X_test, y_test)])
        else:
            model.fit(X_train_res, y_train_res)

        train_time = time.perf_counter() - t0

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        lat    = _infer_latency(model, X_test[:10])

        mets   = _eval(y_test, y_pred, y_prob, feat_names, model, train_time)
        mets["latency_ms"] = lat
        mets["model_name"] = name

        comparison[name] = mets
        leaderboard.append({
            "rank":       0,
            "model":      name,
            "pr_auc":     mets["pr_auc"],
            "roc_auc":    mets["roc_auc"],
            "f1_score":   mets["f1_score"],
            "precision":  mets["precision"],
            "recall":     mets["recall"],
            "mcc":        mets["mcc"],
            "g_mean":     mets["g_mean"],
            "train_time": mets["train_time_s"],
            "latency_ms": lat,
        })

        safe_name = name.replace(" ","_").lower()
        joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}_model.joblib"))
        print(f"  PR-AUC: {mets['pr_auc']}  |  F1: {mets['f1_score']}  |  Lat: {lat}ms")

        if name == "XGBoost":
            primary_xgb     = model
            primary_metrics = mets

    # ── Isolation Forest (Unsupervised) ────────────────────────────────────────
    print("\n[training] Isolation Forest (Unsupervised Anomaly Detection) ...")
    iso = IsolationForest(n_estimators=200, contamination=pos/(neg+pos),
                          random_state=random_state, n_jobs=-1)
    iso.fit(X_train_res)
    iso_scores = -iso.score_samples(X_test)
    iso_scores = (iso_scores - iso_scores.min()) / \
                  (iso_scores.max() - iso_scores.min() + 1e-9)
    iso_pred   = (iso_scores >= 0.5).astype(int)
    iso_pc, iso_rc, _ = precision_recall_curve(y_test, iso_scores)
    iso_pr  = round(auc(iso_rc, iso_pc), 4)
    iso_roc = round(roc_auc_score(y_test, iso_scores), 4)
    joblib.dump(iso, os.path.join(MODELS_DIR, "isolation_forest_model.joblib"))
    comparison["Isolation Forest"] = {
        "pr_auc": iso_pr, "roc_auc": iso_roc,
        "f1_score": round(f1_score(y_test, iso_pred, zero_division=0),4),
        "model_name": "Isolation Forest",
        "note": "Unsupervised — no class labels used during training",
    }
    leaderboard.append({
        "rank":0, "model":"Isolation Forest",
        "pr_auc":iso_pr, "roc_auc":iso_roc,
        "f1_score":round(f1_score(y_test,iso_pred,zero_division=0),4),
        "precision":"–","recall":"–","mcc":"–","g_mean":"–",
        "train_time":"–","latency_ms":"–",
    })
    print(f"  PR-AUC: {iso_pr}  |  ROC-AUC: {iso_roc}")

    # ── Rank leaderboard ───────────────────────────────────────────────────────
    leaderboard.sort(key=lambda x: x["pr_auc"] if isinstance(x["pr_auc"], float) else 0,
                     reverse=True)
    for i, row in enumerate(leaderboard):
        row["rank"] = i + 1

    # ── Save artifacts ─────────────────────────────────────────────────────────
    joblib.dump(comparison, os.path.join(MODELS_DIR, "comparison.joblib"))
    joblib.dump(leaderboard, os.path.join(MODELS_DIR, "leaderboard.joblib"))
    joblib.dump({**primary_metrics, "feature_names": feat_names},
                os.path.join(MODELS_DIR, "metrics.joblib"))
    # Keep xgb_model.joblib as primary
    joblib.dump(primary_xgb, os.path.join(MODELS_DIR, "xgb_model.joblib"))

    # ── Final Report ───────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("LEADERBOARD — ALL MODELS RANKED BY PR-AUC")
    print("="*65)
    print(f"{'Rank':<6}{'Model':<22}{'PR-AUC':<10}{'ROC-AUC':<10}{'F1':<8}{'Lat(ms)'}")
    print("-"*65)
    for r in leaderboard:
        print(f"{r['rank']:<6}{r['model']:<22}{str(r['pr_auc']):<10}"
              f"{str(r['roc_auc']):<10}{str(r['f1_score']):<8}{r['latency_ms']}")
    print("="*65)
    print(f"[training] All 8 models saved -> {MODELS_DIR}")

    return primary_xgb, primary_metrics


if __name__ == "__main__":
    train_model()
