"""
app.py  —  FraudSentinel · World-Class Flask API
=================================================
Group 1 Capstone: Abhishek Kumar Saroj, Aman Kumar, Amit, Ankit
"""

import os, sys, json, random, io, csv, uuid, requests, time
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import shap
import plotly, plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, Response
from src.database import (
    insert_tx, get_history, search_by_id, get_db_stats,
    get_active_model, set_active_model, insert_batch
)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")
CSV_PATH   = os.path.join(DATA_DIR, "creditcard_synthetic.csv")

app = Flask(__name__)

# ── Auto-train on startup ──────────────────────────────────────────────────────
def ensure_model():
    if not os.path.exists(os.path.join(MODELS_DIR, "xgb_model.joblib")):
        print("[startup] Model not found — training now (this takes ~2 min)...")
        from src.model_training import train_model
        train_model()

ensure_model()

# ── Load artifacts ─────────────────────────────────────────────────────────────
xgb_model  = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
metrics    = joblib.load(os.path.join(MODELS_DIR, "metrics.joblib"))
feat_names = metrics.get("feature_names",
                         ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"])
explainer  = shap.TreeExplainer(xgb_model)
df_data    = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else None

# Load comparison & leaderboard
_comp_path = os.path.join(MODELS_DIR, "comparison.joblib")
comparison = joblib.load(_comp_path) if os.path.exists(_comp_path) else {}
_lb_path   = os.path.join(MODELS_DIR, "leaderboard.joblib")
leaderboard_data = joblib.load(_lb_path) if os.path.exists(_lb_path) else []

# Load ALL 8 models for consensus
MODEL_FILES = {
    "XGBoost":             "xgb_model.joblib",
    "LightGBM":            "lightgbm_model.joblib",
    "CatBoost":            "catboost_model.joblib",
    "Random Forest":       "random_forest_model.joblib",
    "Logistic Regression": "logistic_regression_model.joblib",
    "SVM":                 "svm_model.joblib",
    "MLP Neural Network":  "mlp_neural_network_model.joblib",
}
all_models = {}
for mname, mfile in MODEL_FILES.items():
    mpath = os.path.join(MODELS_DIR, mfile)
    if os.path.exists(mpath):
        all_models[mname] = joblib.load(mpath)

# In-memory live feed storage
live_feed   = []
live_stats  = {"total":0,"fraud":0,"legit":0,"total_amount":0.0}

# System Logs for Terminal
system_logs = []
def add_sys_log(msg, log_type='info'):
    time_str = datetime.now().strftime("%H:%M:%S")
    system_logs.append({"msg": msg, "type": log_type, "time": time_str})
    if len(system_logs) > 100:
        system_logs.pop(0)

# Real metrics tracking
SERVER_START_TIME = time.time()
transaction_timestamps = []

def record_transaction():
    now = time.time()
    transaction_timestamps.append(now)
    # Keep only the last 10 seconds of timestamps for TPS calculation
    while transaction_timestamps and transaction_timestamps[0] < now - 10:
        transaction_timestamps.pop(0)

# Training data stats for drift detection
_drift_means = None
if df_data is not None:
    _drift_means = df_data[feat_names].mean().to_dict()

# ── AI Reasoner ───────────────────────────────────────────────────────────────
def generate_ai_reason(shap_vals, prob, pred, amount, row):
    top = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    label = "FRAUDULENT" if pred==1 else "LEGITIMATE"
    color = "#ff003c" if pred==1 else "#00ff9f"
    parts = []
    for feat, val in top:
        direction = "elevated" if val > 0 else "suppressed"
        if feat == "Amount":
            parts.append(f"Amount (${amount:.2f}) is {direction} risk")
        elif feat == "Time":
            parts.append(f"Transaction timing shows {direction} anomaly")
        else:
            parts.append(f"{feat} signal is {direction} (SHAP={val:+.3f})")
    reason = f"Transaction flagged as <strong style='color:{color}'>{label}</strong> "
    reason += f"with {prob*100:.1f}% confidence. "
    if pred==1:
        reason += "Key indicators: " + "; ".join(parts) + ". "
        reason += "Pattern consistent with known fraud signatures in PCA feature space."
    else:
        reason += "No significant anomalies detected. " + "; ".join(parts) + "."
    return reason


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — Plotly common theme
# ══════════════════════════════════════════════════════════════════════════════
def cyber_layout(**kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,13,13,0.95)",
        font=dict(family="Share Tech Mono,monospace", color="#00ff9f", size=11),
        margin=dict(l=10,r=10,t=44,b=30),
    )
    base.update(kwargs)
    return base


def grid(color="#00ff9f", opacity=0.08):
    return dict(gridcolor=f"rgba(0,255,159,{opacity})")


def axis_style(title="", **kw):
    return dict(title=title, color="#00c8ff",
                gridcolor="rgba(0,255,159,0.08)",
                zerolinecolor="rgba(0,255,159,0.3)", **kw)


def _chart(fig): return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    cm  = metrics.get("confusion_matrix",[[0,0],[0,0]])
    tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    total = tn+fp+fn+tp
    prec  = round(tp/(tp+fp)*100,1) if (tp+fp) else 0
    rec   = round(tp/(tp+fn)*100,1) if (tp+fn) else 0

    # Confusion matrix
    fig_cm = go.Figure(go.Heatmap(
        z=[[tn,fp],[fn,tp]],
        x=["Predicted: Clear","Predicted: Threat"],
        y=["Actual: Clear","Actual: Threat"],
        text=[[f"TN: {tn:,}",f"FP: {fp:,}"],[f"FN: {fn:,}",f"TP: {tp:,}"]],
        texttemplate="%{text}",
        textfont={"size":14,"color":"#00ff9f","family":"Share Tech Mono"},
        colorscale=[[0,"#000d0d"],[0.5,"rgba(0,120,80,.6)"],[1,"rgba(0,255,159,.4)"]],
        showscale=False,
    ))
    fig_cm.update_layout(**cyber_layout(height=240,
        xaxis=axis_style(), yaxis=axis_style()))

    # Gauge — PR-AUC
    pr_val = float(metrics.get("pr_auc",0))
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pr_val*100,
        number=dict(suffix="%", font=dict(family="Orbitron,monospace",
                    color="#00ff9f", size=28)),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor="#00c8ff",
                      tickfont=dict(color="#00c8ff",size=9)),
            bar=dict(color="#00ff9f", thickness=0.25),
            bgcolor="rgba(0,13,13,0.9)",
            borderwidth=1, bordercolor="rgba(0,255,159,0.3)",
            steps=[
                dict(range=[0,60],  color="rgba(255,0,60,0.15)"),
                dict(range=[60,80], color="rgba(255,165,0,0.12)"),
                dict(range=[80,100],color="rgba(0,255,159,0.08)"),
            ],
            threshold=dict(line=dict(color="#ff003c",width=2),
                           thickness=0.75, value=80),
        ),
        domain=dict(x=[0,1], y=[0,1]),
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#00ff9f"),
        height=200, margin=dict(l=20,r=20,t=20,b=10),
    )

    kpis = dict(
        total=f"{total:,}", fraud=f"{tp+fn:,}",
        pr_auc=f"{metrics.get('pr_auc',0):.3f}",
        prec=f"{prec}%", recall=f"{rec}%",
        mcc=f"{metrics.get('mcc',0):.3f}",
        roc=f"{metrics.get('roc_auc',0):.3f}",
    )
    return render_template("index.html", kpis=kpis,
                           cm_json=_chart(fig_cm),
                           gauge_json=_chart(fig_gauge),
                           feat_names=feat_names)


@app.route("/eda")
def eda():
    if df_data is None:
        return render_template("eda.html", error=True)

    counts  = df_data["Class"].value_counts()
    # Donut
    fig_d = go.Figure(go.Pie(
        labels=["Legitimate","Fraud"],
        values=[int(counts.get(0,0)), int(counts.get(1,0))],
        hole=0.62,
        marker=dict(colors=["#00ff9f","#ff003c"],
                    line=dict(color="#050f0f",width=3)),
        textinfo="label+percent",
        textfont=dict(size=11,color="#050f0f",family="Share Tech Mono"),
    ))
    fig_d.update_layout(**cyber_layout(height=300,
        annotations=[dict(text="CLASS<br>SPLIT",x=0.5,y=0.5,showarrow=False,
                          font=dict(size=10,color="#00c8ff",family="Share Tech Mono"))],
        showlegend=True,
        legend=dict(font=dict(color="#00ff9f"),bgcolor="rgba(0,0,0,0)")))

    # Amount histogram
    fig_a = go.Figure()
    for cls,color,label in [(0,"#00ff9f","LEGITIMATE"),(1,"#ff003c","THREAT")]:
        fig_a.add_trace(go.Histogram(
            x=df_data.loc[df_data["Class"]==cls,"Amount"].clip(upper=2000),
            name=label, marker_color=color, opacity=0.7, nbinsx=55,
        ))
    fig_a.update_layout(**cyber_layout(height=280, barmode="overlay",
        xaxis=axis_style("Amount (USD)"), yaxis=axis_style("Frequency"),
        legend=dict(font=dict(color="#00ff9f"),bgcolor="rgba(0,0,0,.5)")))

    # 3D PCA scatter
    pca_cols = [c for c in df_data.columns if c.startswith("V")][:3]
    sample3d = df_data.sample(n=min(800,len(df_data)), random_state=42)
    fig_3d = go.Figure()
    for cls,color,label in [(0,"#00ff9f","Legitimate"),(1,"#ff003c","Fraud")]:
        sub = sample3d[sample3d["Class"]==cls]
        fig_3d.add_trace(go.Scatter3d(
            x=sub[pca_cols[0]], y=sub[pca_cols[1]], z=sub[pca_cols[2]],
            mode="markers", name=label,
            marker=dict(size=3, color=color, opacity=0.7,
                        line=dict(width=0)),
        ))
    fig_3d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#00ff9f",size=10),
        height=420, margin=dict(l=0,r=0,t=30,b=0),
        scene=dict(
            bgcolor="rgba(0,13,13,0.95)",
            xaxis=dict(title=pca_cols[0],gridcolor="rgba(0,255,159,.15)",
                       color="#00c8ff"),
            yaxis=dict(title=pca_cols[1],gridcolor="rgba(0,255,159,.15)",
                       color="#00c8ff"),
            zaxis=dict(title=pca_cols[2],gridcolor="rgba(0,255,159,.15)",
                       color="#00c8ff"),
        ),
        legend=dict(font=dict(color="#00ff9f"),bgcolor="rgba(0,0,0,.5)"),
    )

    # Box plot
    fig_b = go.Figure()
    for cls,label,color in [(0,"LEGITIMATE","#00ff9f"),(1,"THREAT","#ff003c")]:
        sub = df_data[df_data["Class"]==cls]
        for feat in ["V1","V2","V3","V4","V10","V12","V14","V17"]:
            fig_b.add_trace(go.Box(
                y=sub[feat], name=feat, legendgroup=label,
                legendgrouptitle=dict(text=label) if feat=="V1" else None,
                marker_color=color, line_color=color, boxmean=True, opacity=0.75,
            ))
    fig_b.update_layout(**cyber_layout(height=360, boxmode="group",
        xaxis=axis_style("PCA Component"), yaxis=axis_style("Value"),
        legend=dict(font=dict(color="#00ff9f"),bgcolor="rgba(0,0,0,.5)")))

    # Correlation heatmap
    pca_c = [c for c in df_data.columns if c.startswith("V")][:14]
    corr  = df_data[pca_c+["Amount","Class"]].corr()
    fig_c = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0,"#ff003c"],[0.5,"#050f0f"],[1,"#00ff9f"]],
        zmid=0, showscale=True,
        colorbar=dict(tickfont=dict(color="#00ff9f",family="Share Tech Mono")),
    ))
    fig_c.update_layout(**cyber_layout(height=440,
        xaxis=dict(tickangle=-45,tickfont=dict(color="#00c8ff",size=9)),
        yaxis=dict(autorange="reversed",tickfont=dict(color="#00c8ff",size=9))))

    # Time distribution
    fig_t = go.Figure()
    for cls,color,label in [(0,"rgba(0,255,159,0.4)","LEGITIMATE"),
                             (1,"rgba(255,0,60,0.6)","THREAT")]:
        fig_t.add_trace(go.Histogram(
            x=df_data.loc[df_data["Class"]==cls,"Time"],
            name=label, marker_color=color, nbinsx=48, opacity=0.75,
        ))
    fig_t.update_layout(**cyber_layout(height=260, barmode="overlay",
        xaxis=axis_style("Time (seconds)"), yaxis=axis_style("Count"),
        legend=dict(font=dict(color="#00ff9f"),bgcolor="rgba(0,0,0,.5)")))

    stats = dict(total=len(df_data), legit=int((df_data["Class"]==0).sum()),
                 fraud=int((df_data["Class"]==1).sum()),
                 rate=f"{df_data['Class'].mean()*100:.3f}%",
                 avg_amt=f"${df_data['Amount'].mean():.2f}",
                 max_amt=f"${df_data['Amount'].max():.2f}")

    charts = dict(
        donut=_chart(fig_d), amt=_chart(fig_a),
        scatter3d=_chart(fig_3d), box=_chart(fig_b),
        corr=_chart(fig_c), time=_chart(fig_t),
    )
    return render_template("eda.html", stats=stats, charts=charts, error=False)


@app.route("/metrics-page")
def metrics_page():
    cm   = metrics.get("confusion_matrix",[[0,0],[0,0]])
    tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec  = tp/(tp+fn) if (tp+fn) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0

    # Radar
    cats = ["PR-AUC","ROC-AUC","F1","PRECISION","RECALL","G-MEAN"]
    vals = [float(metrics.get("pr_auc",0)), float(metrics.get("roc_auc",0)),
            round(f1,4), round(prec,4), round(rec,4),
            float(metrics.get("g_mean",0))]
    fig_r = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
        line=dict(color="#00ff9f",width=2),
        fillcolor="rgba(0,255,159,0.08)",
    ))
    fig_r.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono",color="#00ff9f",size=10),
        height=300, margin=dict(l=20,r=20,t=20,b=20),
        polar=dict(bgcolor="rgba(0,13,13,0.95)",
                   radialaxis=dict(visible=True,range=[0,1],
                                   gridcolor="rgba(0,255,159,.15)",
                                   tickfont=dict(color="#00c8ff",size=8)),
                   angularaxis=dict(gridcolor="rgba(0,255,159,.12)",
                                    tickfont=dict(color="#00ff9f",size=9))),
        showlegend=False)

    # PR curve
    pc = metrics.get("precision_curve",[])
    rc = metrics.get("recall_curve",[])
    fig_pr = go.Figure()
    if pc and rc:
        fig_pr.add_trace(go.Scatter(x=rc, y=pc, mode="lines", name="XGBoost",
            line=dict(color="#00ff9f",width=2.5),
            fill="tozeroy", fillcolor="rgba(0,255,159,0.06)"))
        fig_pr.add_hline(y=float(metrics.get("pr_auc",0)), line_dash="dot",
                         line_color="#ff003c",
                         annotation_text=f"PR-AUC = {metrics.get('pr_auc',0):.4f}",
                         annotation_font=dict(color="#ff003c"))
    fig_pr.update_layout(**cyber_layout(height=280,
        xaxis=axis_style("Recall",range=[0,1]),
        yaxis=axis_style("Precision",range=[0,1])))

    # Feature importance
    imp  = xgb_model.feature_importances_
    fi   = pd.DataFrame({"Feature":feat_names,"Importance":imp})
    fi   = fi.nlargest(20,"Importance").sort_values("Importance")
    fig_fi = go.Figure(go.Bar(
        x=fi["Importance"], y=fi["Feature"], orientation="h",
        marker=dict(color=fi["Importance"],
                    colorscale=[[0,"rgba(0,160,100,.5)"],[1,"#00ff9f"]],
                    showscale=False),
    ))
    fig_fi.update_layout(**cyber_layout(height=460,
        xaxis=axis_style("Importance (Gain)"),
        yaxis=axis_style()))

    # Model comparison bar chart
    model_names = list(comparison.keys())
    comp_metrics = ["pr_auc","roc_auc","f1_score","precision","recall"]
    fig_comp = go.Figure()
    palette = {"PR-AUC":"#00ff9f","ROC-AUC":"#00c8ff","F1":"#a78bfa",
               "Precision":"#f59e0b","Recall":"#ff003c"}
    for met, col in zip(comp_metrics, list(palette.values())):
        fig_comp.add_trace(go.Bar(
            name=met.replace("_"," ").upper(),
            x=model_names,
            y=[comparison[m].get(met,0) if m in comparison else 0
               for m in model_names],
            marker_color=col,
        ))
    fig_comp.update_layout(**cyber_layout(height=300, barmode="group",
        xaxis=axis_style(), yaxis=axis_style("Score",range=[0,1.05]),
        legend=dict(font=dict(color="#00ff9f"),bgcolor="rgba(0,0,0,.5)")))

    table = [
        {"metric":"PR-AUC",     "value":metrics.get("pr_auc","–"),      "note":"Gold standard (imbalanced)"},
        {"metric":"ROC-AUC",    "value":metrics.get("roc_auc","–"),     "note":"Overall discriminative power"},
        {"metric":"F1-Score",   "value":round(f1,4),                    "note":"Precision-Recall balance"},
        {"metric":"Precision",  "value":round(prec,4),                  "note":"False alarm control"},
        {"metric":"Recall",     "value":round(rec,4),                   "note":"Threat interception rate"},
        {"metric":"Specificity","value":metrics.get("specificity","–"), "note":"True negative rate"},
        {"metric":"MCC",        "value":metrics.get("mcc","–"),         "note":"Balanced quality (−1 to +1)"},
        {"metric":"G-Mean",     "value":metrics.get("g_mean","–"),      "note":"Sensitivity × Specificity"},
    ]
    cm_detail = {"tn":tn,"fp":fp,"fn":fn,"tp":tp}
    charts    = dict(radar=_chart(fig_r), pr=_chart(fig_pr),
                     fi=_chart(fig_fi), comp=_chart(fig_comp))
    comp_table = [{"model":m,"pr_auc":d.get("pr_auc",0),
                   "roc_auc":d.get("roc_auc",0),"f1":d.get("f1_score",0),
                   "precision":d.get("precision",0),"recall":d.get("recall",0)}
                  for m,d in comparison.items()]

    return render_template("metrics.html", charts=charts, table_data=table,
                           cm_detail=cm_detail, comp_table=comp_table)


@app.route("/xai")
def xai():
    return render_template("xai.html", feat_names=feat_names)


@app.route("/simulate")
def simulate():
    return render_template("simulate.html")


# ══════════════════════════════════════════════════════════════════════════════
#  REST API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def api_predict():
    record_transaction()
    data      = request.get_json()
    row       = {f: float(data.get(f, 0.0)) for f in feat_names}
    X_in      = pd.DataFrame([row])[feat_names]
    prob      = float(xgb_model.predict_proba(X_in)[0,1])
    threshold = float(data.get("threshold", 0.5))
    pred      = int(prob>=threshold)
    add_sys_log(f"Inference request processed. Risk Score: {prob*100:.2f}%", "warn" if pred==1 else "info")
    return jsonify({"probability": round(prob,4), "prediction": pred})


@app.route("/api/shap", methods=["POST"])
def api_shap():
    data  = request.get_json()
    row   = {f: float(data.get(f, 0.0)) for f in feat_names}
    X_in  = pd.DataFrame([row])[feat_names]
    sv    = explainer.shap_values(X_in)
    top12 = sorted(zip(feat_names, [float(v) for v in sv[0]]),
                   key=lambda x: abs(x[1]), reverse=True)[:12]
    return jsonify({
        "shap_values": dict(top12),
        "base_value":  round(float(explainer.expected_value),5),
        "all_values":  {f:round(float(v),5) for f,v in zip(feat_names,sv[0])},
    })


@app.route("/api/live-tx")
def api_live_tx():
    """Stream a real transaction from the dataset."""
    record_transaction()
    if df_data is None or df_data.empty:
        return jsonify({"error": "Real dataset not found. Cannot stream real transactions."}), 500
        
    # Pick a random row from the real dataset
    real_row = df_data.sample(n=1).iloc[0]
    amount   = float(real_row["Amount"])
    row      = {f: float(real_row.get(f, 0.0)) for f in feat_names}

    X_in = pd.DataFrame([row])[feat_names]
    prob = float(xgb_model.predict_proba(X_in)[0,1])
    pred = int(prob >= 0.5)

    tx_id = f"TX-{rng.integers(100000,999999)}"
    entry = {
        "id":     tx_id,
        "amount": amount,
        "prob":   round(prob,4),
        "pred":   pred,
        "time":   datetime.now().strftime("%H:%M:%S"),
        "type":   random.choice(["TRANSFER","PAYMENT","WITHDRAWAL","DEPOSIT"]),
    }

    # Update live stats
    live_stats["total"]        += 1
    live_stats["total_amount"] += amount
    if pred == 1:
        live_stats["fraud"] += 1
        add_sys_log(f"Live Threat Intercepted: {tx_id} (${amount:.2f})", "err")
    else:
        live_stats["legit"] += 1

    live_feed.insert(0, entry)
    if len(live_feed) > 50:
        live_feed.pop()

    return jsonify(entry)


@app.route("/api/live-stats")
def api_live_stats():
    return jsonify({**live_stats,
                    "fraud_rate": round(live_stats["fraud"] / max(live_stats["total"],1)*100, 2),
                    "avg_amount": round(live_stats["total_amount"] / max(live_stats["total"],1), 2)})


@app.route("/api/live-feed")
def api_live_feed():
    return jsonify(live_feed[:20])


@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    record_transaction()
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}), 400
    f      = request.files["file"]
    df_up  = pd.read_csv(f)
    threshold = float(request.form.get("threshold","0.5"))
    for col in feat_names:
        if col not in df_up.columns:
            df_up[col] = 0.0
    X_up  = df_up[feat_names]
    probs = xgb_model.predict_proba(X_up)[:,1]
    preds = (probs >= threshold).astype(int)
    df_up["THREAT_SCORE"] = probs.round(4)
    df_up["PREDICTION"]   = np.where(preds==1,"FRAUD","LEGITIMATE")
    fc    = int(preds.sum())
    tc    = len(df_up)
    rows  = df_up[["THREAT_SCORE","PREDICTION"]+feat_names[:5]].head(100).to_dict("records")
    add_sys_log(f"Batch prediction finished: {tc} rows scanned, {fc} threats found.", "warn" if fc > 0 else "info")
    return jsonify({"total":tc, "fraud":fc, "legit":tc-fc,
                    "fraud_rate":round(fc/tc*100,2) if tc else 0, "rows":rows})


@app.route("/api/export-csv")
def api_export_csv():
    """Export live feed as downloadable CSV."""
    if not live_feed:
        return "No data", 404
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["id","time","type","amount","prob","pred"])
    writer.writeheader()
    writer.writerows(live_feed)
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition":"attachment;filename=fraud_report.csv"}
    )


@app.route("/api/model-compare")
def api_model_compare():
    return jsonify(comparison)


@app.route("/api/summary")
def api_summary():
    """Live summary for dashboard auto-refresh."""
    return jsonify({
        "pr_auc":  metrics.get("pr_auc",0),
        "roc_auc": metrics.get("roc_auc",0),
        "mcc":     metrics.get("mcc",0),
        "live":    live_stats,
    })

@app.route("/api/system-logs")
def api_system_logs():
    global system_logs
    logs_to_send = list(system_logs)
    system_logs.clear()
    return jsonify(logs_to_send)

@app.route("/api/health")
def api_health():
    """Returns real server uptime and actual TPS."""
    now = time.time()
    uptime = int(now - SERVER_START_TIME)
    
    # Calculate TPS over the last 10 seconds
    while transaction_timestamps and transaction_timestamps[0] < now - 10:
        transaction_timestamps.pop(0)
    tps = len(transaction_timestamps) / 10.0
    
    return jsonify({
        "uptime": uptime,
        "tps": round(tps, 1)
    })

@app.route("/api/simulate-batch")
def api_simulate_batch():
    """Pull 50 real transactions from the dataset for the simulator page."""
    if df_data is None or df_data.empty:
        return jsonify({"error": "Real dataset not found."}), 500
        
    results = []
    sample_df = df_data.sample(n=min(50, len(df_data)))
    
    for idx, real_row in sample_df.iterrows():
        amount = float(real_row["Amount"])
        row = {f: float(real_row.get(f, 0.0)) for f in feat_names}
        X_in = pd.DataFrame([row])[feat_names]
        prob = float(xgb_model.predict_proba(X_in)[0,1])
        results.append({
            "amount": amount, "prob": round(prob,4),
            "pred": int(prob>=0.5),
            "time": datetime.now().strftime("%H:%M:%S"),
        })
    return jsonify(results)


# ── NEW PAGE ROUTES ───────────────────────────────────────────────────────────

@app.route("/leaderboard")
def leaderboard():
    return render_template("leaderboard.html", leaderboard=leaderboard_data,
                           active=get_active_model())

@app.route("/history")
def history():
    page = int(request.args.get("page", 1))
    pred = request.args.get("pred", None)
    pred = int(pred) if pred in ["0","1"] else None
    rows, total = get_history(page=page, per_page=50, prediction=pred)
    db_stats = get_db_stats()
    return render_template("history.html", rows=rows, total=total,
                           page=page, per_page=50, db_stats=db_stats)

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/adversarial")
def adversarial():
    return render_template("adversarial.html", feat_names=feat_names)

@app.route("/api-docs")
def api_docs():
    return render_template("api_docs.html")

# ── NEW API ENDPOINTS ─────────────────────────────────────────────────────────

@app.route("/api/consensus", methods=["POST"])
def api_consensus():
    data      = request.get_json()
    threshold = float(data.get("threshold", 0.5))
    row       = {f: float(data.get(f, 0.0)) for f in feat_names}
    X_in      = pd.DataFrame([row])[feat_names]
    votes, fraud_votes = [], 0
    for mname, model in all_models.items():
        try:
            prob = float(model.predict_proba(X_in)[0,1])
            pred = int(prob >= threshold)
            if pred == 1: fraud_votes += 1
            votes.append({"model": mname, "prob": round(prob,4), "pred": pred})
        except Exception:
            pass
    total_v   = len(votes)
    consensus = "HIGH CONFIDENCE FRAUD" if fraud_votes >= total_v*0.75 \
           else "SUSPICIOUS — REVIEW" if fraud_votes >= total_v*0.4 \
           else "LIKELY LEGITIMATE"
    color     = "#ff003c" if fraud_votes >= total_v*0.75 \
           else "#f59e0b" if fraud_votes >= total_v*0.4 \
           else "#00ff9f"
    return jsonify({"votes": votes, "fraud_votes": fraud_votes,
                    "total_models": total_v, "consensus": consensus,
                    "consensus_color": color,
                    "fraud_pct": round(fraud_votes/max(total_v,1)*100, 1)})


@app.route("/api/ai-reason", methods=["POST"])
def api_ai_reason():
    data      = request.get_json()
    row       = {f: float(data.get(f, 0.0)) for f in feat_names}
    X_in      = pd.DataFrame([row])[feat_names]
    prob      = float(xgb_model.predict_proba(X_in)[0,1])
    threshold = float(data.get("threshold", 0.5))
    pred      = int(prob >= threshold)
    sv        = explainer.shap_values(X_in)
    shap_d    = {f: round(float(v),4) for f,v in zip(feat_names, sv[0])}
    reason    = generate_ai_reason(shap_d, prob, pred, row["Amount"], row)
    return jsonify({"reason": reason, "probability": round(prob,4), "prediction": pred})


@app.route("/api/leaderboard")
def api_leaderboard():
    return jsonify(leaderboard_data)


@app.route("/api/active-model", methods=["POST"])
def api_set_model():
    name = request.get_json().get("model","XGBoost")
    if name in all_models or name == "XGBoost":
        set_active_model(name)
        add_sys_log(f"Engine Switched: {name} is now ACTIVE.", "sys")
        return jsonify({"status": "ok", "active": name})
    return jsonify({"error": "Model not found"}), 404


@app.route("/api/history")
def api_history_json():
    page = int(request.args.get("page",1))
    pred = request.args.get("pred",None)
    pred = int(pred) if pred in ["0","1"] else None
    rows, total = get_history(page=page, per_page=50, prediction=pred)
    return jsonify({"rows": rows, "total": total})


@app.route("/api/history/search")
def api_history_search():
    tx_id = request.args.get("id","")
    result = search_by_id(tx_id)
    return jsonify(result if result else {"error": "Not found"})


@app.route("/api/drift")
def api_drift():
    if not _drift_means or not live_feed:
        return jsonify({"status": "insufficient_data", "psi": 0, "alert": False})
    live_amounts = [tx["amount"] for tx in live_feed[-20:]]
    train_mean   = _drift_means.get("Amount", 100)
    live_mean    = np.mean(live_amounts) if live_amounts else train_mean
    psi          = abs(live_mean - train_mean) / max(train_mean, 1)
    alert        = psi > 0.25
    return jsonify({"psi": round(psi,4), "alert": alert,
                    "train_mean": round(train_mean,2),
                    "live_mean": round(live_mean,2),
                    "message": "ALERT: Model Accuracy may degrade due to Concept Drift!" if alert
                               else "Model distribution stable."})


@app.route("/api/adversarial", methods=["POST"])
def api_adversarial():
    data  = request.get_json()
    row   = {f: float(data.get(f, 0.0)) for f in feat_names}
    X_in  = pd.DataFrame([row])[feat_names]
    prob  = float(xgb_model.predict_proba(X_in)[0,1])
    pred  = int(prob >= 0.5)
    # Check if this looks like an adversarial attempt
    v14   = row.get("V14", 0)
    v12   = row.get("V12", 0)
    is_adv = pred==0 and (abs(v14) > 3 or abs(v12) > 3)
    return jsonify({
        "probability": round(prob,4), "prediction": pred,
        "is_adversarial": is_adv,
        "message": "ADVERSARIAL ATTACK DETECTED — Suspicious feature pattern evaded classifier."
                   if is_adv else
                   ("FRAUD CAUGHT — AI cannot be fooled!" if pred==1 else
                    "Transaction appears legitimate.")
    })


@app.route("/api/export-pdf")
def api_export_pdf():
    try:
        from fpdf import FPDF
        db_stats = get_db_stats()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(0, 200, 100)
        pdf.cell(0, 12, "FraudSentinel v3.0 — Audit Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "Model Performance (XGBoost Primary)", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for k, v in [("PR-AUC", metrics.get("pr_auc")),
                     ("ROC-AUC", metrics.get("roc_auc")),
                     ("MCC", metrics.get("mcc")),
                     ("G-Mean", metrics.get("g_mean"))]:
            pdf.cell(0, 7, f"  {k}: {v}", ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Live Session Statistics", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"  Total Scanned: {db_stats['total']:,}", ln=True)
        pdf.cell(0, 7, f"  Threats Detected: {db_stats['fraud']:,}", ln=True)
        pdf.cell(0, 7, f"  Fraud Rate: {db_stats['fraud_rate']}%", ln=True)
        pdf.cell(0, 7, f"  Total Amount: ${db_stats['total_amount']:,.2f}", ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Model Leaderboard", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for r in leaderboard_data[:8]:
            pdf.cell(0, 6,
                f"  #{r['rank']} {r['model']} — PR-AUC: {r['pr_auc']}  F1: {r['f1_score']}  Latency: {r['latency_ms']}ms",
                ln=True)
        pdf.ln(6)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(120,120,120)
        pdf.cell(0, 6, "Group 1 Capstone — Advanced Machine Learning 2026", ln=True, align="C")
        out = io.BytesIO(pdf.output())
        return Response(out.getvalue(), mimetype="application/pdf",
                        headers={"Content-Disposition": "attachment;filename=FraudSentinel_Report.pdf"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload-file", methods=["POST"])
def api_upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f         = request.files["file"]
    threshold = float(request.form.get("threshold", 0.5))
    fname     = f.filename.lower()
    try:
        if fname.endswith(".csv"):
            df_up = pd.read_csv(f)
        elif fname.endswith((".xlsx",".xls")):
            df_up = pd.read_excel(f)
        elif fname.endswith(".json"):
            df_up = pd.read_json(f)
        else:
            return jsonify({"error": "Unsupported format. Use CSV, Excel, or JSON."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    for col in feat_names:
        if col not in df_up.columns:
            df_up[col] = 0.0
    X_up  = df_up[feat_names]
    probs = xgb_model.predict_proba(X_up)[:,1]
    preds = (probs >= threshold).astype(int)
    df_up["THREAT_SCORE"] = probs.round(4)
    df_up["PREDICTION"]   = np.where(preds==1,"FRAUD","LEGITIMATE")
    fc  = int(preds.sum())
    tc  = len(df_up)
    scan_id = str(uuid.uuid4())[:8].upper()
    insert_batch(scan_id, f.filename, tc, fc, get_active_model())
    rows = df_up[["THREAT_SCORE","PREDICTION"]+
                  [c for c in feat_names[:4] if c in df_up.columns]].head(100).to_dict("records")
    return jsonify({"scan_id": scan_id, "total": tc, "fraud": fc,
                    "legit": tc-fc,
                    "fraud_rate": round(fc/tc*100,2) if tc else 0,
                    "rows": rows})

@app.route("/api/fetch-external", methods=["POST"])
def api_fetch_external():
    data = request.get_json()
    url = data.get("url")
    api_key = data.get("api_key")
    threshold = float(data.get("threshold", 0.5))

    if not url:
        return jsonify({"error": "No API URL provided"}), 400

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key

    try:
        # Mock fetch or real fetch depending on if it's a real API
        # For safety in this capstone, we will attempt to fetch, 
        # but if it fails, we will generate synthetic "live" external data 
        # to simulate a successful connection for demonstration purposes.
        try:
            r = requests.get(url, headers=headers, timeout=5)
            r.raise_for_status()
            json_data = r.json()
            if isinstance(json_data, dict) and "data" in json_data:
                json_data = json_data["data"]
            df_up = pd.DataFrame(json_data)
        except Exception as e:
            add_sys_log(f"External API Failed: {url} - {str(e)}", "err")
            return jsonify({"error": f"Failed to fetch from {url}. Real API integration requires a valid endpoint. Error: {str(e)}"}), 502

        for col in feat_names:
            if col not in df_up.columns:
                df_up[col] = 0.0
        X_up  = df_up[feat_names]
        probs = xgb_model.predict_proba(X_up)[:,1]
        preds = (probs >= threshold).astype(int)
        df_up["THREAT_SCORE"] = probs.round(4)
        df_up["PREDICTION"]   = np.where(preds==1,"FRAUD","LEGITIMATE")
        fc  = int(preds.sum())
        tc  = len(df_up)
        scan_id = str(uuid.uuid4())[:8].upper()
        insert_batch(scan_id, "EXTERNAL_API_FETCH", tc, fc, get_active_model())
        rows = df_up[["THREAT_SCORE","PREDICTION"]+
                      [c for c in feat_names[:4] if c in df_up.columns]].head(100).to_dict("records")
        return jsonify({"scan_id": scan_id, "total": tc, "fraud": fc,
                        "legit": tc-fc,
                        "fraud_rate": round(fc/tc*100,2) if tc else 0,
                        "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
