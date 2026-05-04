"""
app.py  —  FraudSentinel · World-Class Flask API
=================================================
Group 1 Capstone: Abhishek Kumar Saroj, Aman Kumar, Amit, Ankit
"""

import os, sys, json, random, io, csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
import shap
import plotly, plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

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

# Load comparison if exists
_comp_path = os.path.join(MODELS_DIR, "comparison.joblib")
comparison = joblib.load(_comp_path) if os.path.exists(_comp_path) else {}

# In-memory live feed storage
live_feed   = []
live_stats  = {"total":0,"fraud":0,"legit":0,"total_amount":0.0}


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
    data      = request.get_json()
    row       = {f: float(data.get(f, 0.0)) for f in feat_names}
    X_in      = pd.DataFrame([row])[feat_names]
    prob      = float(xgb_model.predict_proba(X_in)[0,1])
    threshold = float(data.get("threshold", 0.5))
    return jsonify({"probability": round(prob,4), "prediction": int(prob>=threshold)})


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
    """Generate and score a random synthetic transaction."""
    rng     = np.random.default_rng()
    is_fraud = rng.random() < 0.08   # 8% fraud rate for demo excitement
    amount  = float(rng.lognormal(2.0 if is_fraud else 3.5,
                                  1.0 if is_fraud else 1.2))
    amount  = round(min(amount, 25000), 2)
    row     = {f: 0.0 for f in feat_names}
    row["Amount"] = amount
    row["Time"]   = float(rng.integers(0, 172800))

    # PCA features
    if is_fraud:
        shifts = {"V1":-3.0,"V2":3.5,"V3":-4.0,"V4":4.2,"V10":2.3,"V12":-3.1,"V14":-4.5}
        for v,s in shifts.items():
            if v in row: row[v] = float(rng.normal(s, 0.8))
    else:
        for f in feat_names:
            if f.startswith("V"): row[f] = float(rng.normal(0, 1))

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


@app.route("/api/simulate-batch")
def api_simulate_batch():
    """Generate 50 transactions at once for the simulator page."""
    results = []
    rng = np.random.default_rng()
    for _ in range(50):
        is_fraud = rng.random() < 0.08
        amount   = round(float(rng.lognormal(2.0 if is_fraud else 3.5, 1.0)), 2)
        row      = {f:0.0 for f in feat_names}
        row["Amount"] = amount
        row["Time"]   = float(rng.integers(0,172800))
        if is_fraud:
            shifts = {"V1":-3.0,"V2":3.5,"V3":-4.0,"V4":4.2,"V14":-4.5}
            for v,s in shifts.items():
                if v in row: row[v] = float(rng.normal(s,0.8))
        else:
            for f in feat_names:
                if f.startswith("V"): row[f] = float(rng.normal(0,1))
        X_in = pd.DataFrame([row])[feat_names]
        prob = float(xgb_model.predict_proba(X_in)[0,1])
        results.append({
            "amount": amount, "prob": round(prob,4),
            "pred": int(prob>=0.5),
            "time": datetime.now().strftime("%H:%M:%S"),
        })
    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
