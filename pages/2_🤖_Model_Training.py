"""
pages/2_🤖_Model_Training.py  —  CYBER EDITION
FraudSentinel Capstone · Group 1
"""
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
MODELS_DIR = os.path.join(ROOT, "models")

st.set_page_config(page_title="Model Metrics · FraudSentinel CYBER",
                   page_icon="🤖", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');
html,[class*="css"]{font-family:'Rajdhani',sans-serif;color:#00ff9f;}
.stApp{background-color:#000d0d;
       background-image:linear-gradient(rgba(0,255,159,.03) 1px,transparent 1px),
                        linear-gradient(90deg,rgba(0,255,159,.03) 1px,transparent 1px);
       background-size:40px 40px;}
.stApp::before{content:'';position:fixed;top:0;left:0;width:100%;height:100%;
   background:repeating-linear-gradient(0deg,transparent,transparent 2px,
   rgba(0,0,0,.12) 2px,rgba(0,0,0,.12) 4px);pointer-events:none;z-index:9999;}
[data-testid="stSidebar"]{background:#000d0d;border-right:1px solid #00ff9f;}
[data-testid="stSidebar"] *{color:#00ff9f!important;}
.cyber-sec{border-left:3px solid #00ff9f;padding:8px 16px;margin:24px 0 14px;
           background:linear-gradient(90deg,rgba(0,255,159,.08),transparent);}
.cyber-sec h2{font-family:'Orbitron',monospace;color:#00ff9f;font-size:.9rem;
              font-weight:700;margin:0;letter-spacing:.1em;
              text-shadow:0 0 10px rgba(0,255,159,.6);}
hr{border-color:rgba(0,255,159,.2)!important;}
[data-testid="stMetricValue"]{color:#00ff9f!important;font-family:'Orbitron',monospace!important;}
[data-testid="stMetricLabel"]{color:#00c8ff!important;font-family:'Share Tech Mono',monospace!important;}
[data-testid="metric-container"]{background:rgba(0,255,159,.05);
    border:1px solid rgba(0,255,159,.3);border-radius:4px;padding:10px;}
.stDataFrame{border:1px solid rgba(0,255,159,.3)!important;}
</style>""", unsafe_allow_html=True)

PLOTLY_C = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,13,13,0.9)",
                font=dict(family="Share Tech Mono,monospace",color="#00ff9f",size=11),
                margin=dict(l=20,r=20,t=44,b=20))

def sec(t):
    st.markdown(f'<div class="cyber-sec"><h2>// {t}</h2></div>', unsafe_allow_html=True)

st.markdown("""
<div style="padding:28px 0 16px;border-bottom:1px solid rgba(0,255,159,.2);">
<span style="font-family:'Orbitron',monospace;font-size:1.8rem;font-weight:900;
             color:#00ff9f;text-shadow:0 0 15px #00ff9f;">🤖 NEURAL METRICS</span>
<span style="font-family:'Share Tech Mono',monospace;font-size:.8rem;color:#00c8ff;
             margin-left:14px;">MODEL PERFORMANCE ANALYSIS</span></div>""",
unsafe_allow_html=True)

@st.cache_resource(show_spinner="LOADING NEURAL WEIGHTS...")
def load_all():
    mp = os.path.join(MODELS_DIR,"xgb_model.joblib")
    ep = os.path.join(MODELS_DIR,"metrics.joblib")
    if not os.path.exists(mp):
        return None,{}
    return joblib.load(mp), joblib.load(ep) if os.path.exists(ep) else {}

model, metrics = load_all()
if model is None:
    st.error("MODEL NOT FOUND. Run: python -m src.model_training"); st.stop()

feat_names = metrics.get("feature_names",[])
cm         = metrics.get("confusion_matrix",[[0,0],[0,0]])
tn,fp,fn,tp= cm[0][0],cm[0][1],cm[1][0],cm[1][1]
prec = tp/(tp+fp) if (tp+fp) else 0
rec  = tp/(tp+fn) if (tp+fn) else 0
f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0

sec("PERFORMANCE METRICS DATABASE")
col1,col2 = st.columns([1,1])
with col1:
    mdf = pd.DataFrame({
        "METRIC":  ["PR-AUC","ROC-AUC","F1-SCORE","PRECISION",
                    "RECALL","SPECIFICITY","MCC","G-MEAN"],
        "VALUE":   [metrics.get("pr_auc","–"), metrics.get("roc_auc","–"),
                    round(f1,4), round(prec,4), round(rec,4),
                    metrics.get("specificity","–"), metrics.get("mcc","–"),
                    metrics.get("g_mean","–")],
        "NOTES":   ["Gold standard (imbalanced)","Overall discriminative power",
                    "Precision-Recall balance","False alarm control",
                    "Threat interception rate","True negative rate",
                    "Balanced quality (−1 to +1)","Sensitivity × Specificity"],
    })
    st.dataframe(mdf, use_container_width=True, hide_index=True, height=310)

with col2:
    cats = ["PR-AUC","ROC-AUC","F1","PRECISION","RECALL","G-MEAN"]
    vals = [float(metrics.get("pr_auc",0)), float(metrics.get("roc_auc",0)),
            round(f1,4), round(prec,4), round(rec,4),
            float(metrics.get("g_mean",0))]
    vals += [vals[0]]; cats += [cats[0]]
    fig_r = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        line=dict(color="#00ff9f",width=2),
        fillcolor="rgba(0,255,159,0.1)",
        hovertemplate="<b>%{theta}</b>: %{r:.4f}<extra></extra>",
    ))
    fig_r.update_layout(**PLOTLY_C, height=350,
        polar=dict(
            bgcolor="rgba(0,13,13,0.9)",
            radialaxis=dict(visible=True,range=[0,1],
                            gridcolor="rgba(0,255,159,.15)",
                            tickfont=dict(color="#00c8ff",size=8)),
            angularaxis=dict(gridcolor="rgba(0,255,159,.12)",
                             tickfont=dict(color="#00ff9f",size=10)),
        ),
        showlegend=False,
        title=dict(text="// METRICS RADAR SCAN",
                   font=dict(color="#00c8ff",size=12,family="Orbitron,monospace")),
    )
    st.plotly_chart(fig_r, use_container_width=True)

sec("PR-AUC CURVE :: PRECISION-RECALL ANALYSIS")
pc = metrics.get("precision_curve",[])
rc = metrics.get("recall_curve",[])
if pc and rc:
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=rc, y=pc, mode="lines", name="PR CURVE",
        line=dict(color="#00ff9f",width=2.5),
        fill="tozeroy", fillcolor="rgba(0,255,159,0.07)",
        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
    ))
    fig_pr.add_hline(y=float(metrics.get("pr_auc",0)), line_dash="dot",
                     line_color="#ff003c",
                     annotation_text=f"PR-AUC = {metrics.get('pr_auc',0):.4f}",
                     annotation_font=dict(color="#ff003c",family="Share Tech Mono"))
    fig_pr.update_layout(**PLOTLY_C, height=370,
        xaxis=dict(title="RECALL",range=[0,1],gridcolor="rgba(0,255,159,.1)"),
        yaxis=dict(title="PRECISION",range=[0,1],gridcolor="rgba(0,255,159,.1)"),
        title=dict(text="// PRECISION-RECALL CURVE [TEST SET]",
                   font=dict(color="#00c8ff",size=12,family="Orbitron,monospace")),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

sec("FEATURE IMPORTANCE SCAN :: XGBOOST GAIN")
if feat_names:
    imp = model.feature_importances_
    fi  = pd.DataFrame({"Feature":feat_names,"Importance":imp})
    fi  = fi.nlargest(20,"Importance").sort_values("Importance")
    fig_fi = go.Figure(go.Bar(
        x=fi["Importance"], y=fi["Feature"], orientation="h",
        marker=dict(color=fi["Importance"],
                    colorscale=[[0,"rgba(0,180,120,.4)"],[1,"#00ff9f"]],
                    showscale=False),
        hovertemplate="<b>%{y}</b><br>Gain: %{x:.4f}<extra></extra>",
    ))
    fig_fi.update_layout(**PLOTLY_C, height=520,
        xaxis=dict(title="FEATURE IMPORTANCE (GAIN)",gridcolor="rgba(0,255,159,.1)"),
        yaxis=dict(gridcolor="rgba(0,255,159,.06)"),
        title=dict(text="// TOP 20 THREAT INDICATORS",
                   font=dict(color="#00c8ff",size=12,family="Orbitron,monospace")),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

sec("CONFUSION MATRIX :: QUADRANT BREAKDOWN")
c1,c2,c3,c4 = st.columns(4)
c1.metric("TRUE POSITIVES ✓",  f"{tp:,}", help="Fraud correctly intercepted")
c2.metric("TRUE NEGATIVES ✓",  f"{tn:,}", help="Legitimate correctly cleared")
c3.metric("FALSE POSITIVES ⚠", f"{fp:,}", help="Legitimate flagged as threat")
c4.metric("FALSE NEGATIVES 🚨", f"{fn:,}", help="Threats that evaded detection")

st.markdown("---")
st.markdown("""<div style="text-align:center;font-family:'Share Tech Mono',monospace;
font-size:.62rem;color:rgba(0,255,159,.25);padding:8px 0;">
FRAUDSENTINEL · METRICS_MODULE · GROUP_01 · 2026</div>""", unsafe_allow_html=True)
