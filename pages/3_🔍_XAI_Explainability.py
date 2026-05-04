"""
pages/3_🔍_XAI_Explainability.py  —  CYBER EDITION
FraudSentinel Capstone · Group 1
"""
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")
CSV_PATH   = os.path.join(DATA_DIR, "creditcard_synthetic.csv")

st.set_page_config(page_title="XAI · FraudSentinel CYBER", page_icon="🔍", layout="wide")

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
.info-box{background:rgba(0,255,159,.04);border:1px solid rgba(0,255,159,.25);
          border-radius:4px;padding:16px 20px;
          font-family:'Share Tech Mono',monospace;
          color:rgba(0,255,159,.7);font-size:.82rem;line-height:1.7;}
hr{border-color:rgba(0,255,159,.2)!important;}
.stSlider [data-testid="stSlider"]>div>div>div{background:#00ff9f!important;}
label{color:#00c8ff!important;font-family:'Share Tech Mono',monospace!important;}
</style>""", unsafe_allow_html=True)

PLOTLY_C = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,13,13,0.9)",
                font=dict(family="Share Tech Mono,monospace",color="#00ff9f",size=11),
                margin=dict(l=20,r=20,t=44,b=20))

def sec(t):
    st.markdown(f'<div class="cyber-sec"><h2>// {t}</h2></div>', unsafe_allow_html=True)

st.markdown("""
<div style="padding:28px 0 16px;border-bottom:1px solid rgba(0,255,159,.2);">
<span style="font-family:'Orbitron',monospace;font-size:1.8rem;font-weight:900;
             color:#00ff9f;text-shadow:0 0 15px #00ff9f;">🔍 NEURAL DECODE</span>
<span style="font-family:'Share Tech Mono',monospace;font-size:.8rem;color:#00c8ff;
             margin-left:14px;">XAI :: SHAP EXPLAINABILITY ENGINE</span></div>""",
unsafe_allow_html=True)

@st.cache_resource(show_spinner="LOADING NEURAL WEIGHTS...")
def load_model_metrics():
    mp = os.path.join(MODELS_DIR,"xgb_model.joblib")
    ep = os.path.join(MODELS_DIR,"metrics.joblib")
    if not os.path.exists(mp):
        return None,{}
    return joblib.load(mp), joblib.load(ep) if os.path.exists(ep) else {}

@st.cache_data(show_spinner="LOADING DATA FEED...")
def load_data():
    return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else None

@st.cache_data(show_spinner="COMPUTING SHAP MATRIX (30s)...")
def compute_shap(sample_json, feat_names):
    sample = pd.read_json(sample_json)
    mdl    = joblib.load(os.path.join(MODELS_DIR,"xgb_model.joblib"))
    exp    = shap.TreeExplainer(mdl)
    sv     = exp.shap_values(sample)
    return sv, sample, float(exp.expected_value)

model, metrics = load_model_metrics()
if model is None:
    st.error("MODEL NOT FOUND. Run: python -m src.model_training"); st.stop()

feat_names = metrics.get("feature_names",
                         ["Time"]+[f"V{i}" for i in range(1,29)]+["Amount"])
df = load_data()

# ── SHAP Theory ────────────────────────────────────────────────────────────────
sec("PROTOCOL :: SHAP EXPLANATION FRAMEWORK")
st.markdown("""
<div class="info-box">
<b style="color:#00ff9f;">[ SHAP — SHapley Additive exPlanations ]</b><br/>
Rooted in cooperative game theory. Computes marginal contribution of each feature
by evaluating model output across all possible feature coalitions.<br/><br/>
<b style="color:#ff003c;">[ POSITIVE SHAP ]</b> → amplifies threat score → pushes toward FRAUD detection<br/>
<b style="color:#00ff9f;">[ NEGATIVE SHAP ]</b> → suppresses threat score → pushes toward LEGITIMATE<br/><br/>
Global summary = aggregated SHAP across all transactions → reveals which signals
the XGBoost neural network relies on most to classify financial threats.
</div>""", unsafe_allow_html=True)

if df is not None:
    sec("GLOBAL SHAP IMPORTANCE SCAN")
    n_sample = st.slider("SAMPLE SIZE (larger → slower computation)",
                         50, 500, 150, 50)
    df_s = (df[feat_names+["Class"]]
            .sample(n=min(n_sample,len(df)), random_state=42)
            .reset_index(drop=True))
    X_s  = df_s[feat_names]
    y_s  = df_s["Class"]

    sv, X_samp, base_val = compute_shap(X_s.to_json(), feat_names)

    # SHAP bar chart
    mean_abs = np.abs(sv).mean(axis=0)
    fi_df    = pd.DataFrame({"Feature":feat_names,"Mean |SHAP|":mean_abs})
    fi_df    = fi_df.nlargest(20,"Mean |SHAP|").sort_values("Mean |SHAP|")

    fig_bar = go.Figure(go.Bar(
        x=fi_df["Mean |SHAP|"], y=fi_df["Feature"], orientation="h",
        marker=dict(color=fi_df["Mean |SHAP|"],
                    colorscale=[[0,"rgba(0,180,120,.4)"],[1,"#00ff9f"]],
                    showscale=False),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig_bar.update_layout(**PLOTLY_C, height=520,
        xaxis=dict(title="MEAN |SHAP| VALUE",gridcolor="rgba(0,255,159,.1)"),
        yaxis=dict(gridcolor="rgba(0,255,159,.06)"),
        title=dict(text="// TOP 20 THREAT SIGNAL FEATURES",
                   font=dict(color="#00c8ff",size=12,family="Orbitron,monospace")),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # SHAP Beeswarm (matplotlib)
    sec("BEESWARM SCAN :: SHAP DISTRIBUTION")
    st.markdown(
        "<p style='font-family:Share Tech Mono;font-size:.76rem;"
        "color:rgba(0,255,159,.55);'>"
        "> Each node = one transaction. Colour = feature magnitude. "
        "Position = SHAP impact on threat score.</p>",
        unsafe_allow_html=True)

    fig_bee, ax = plt.subplots(figsize=(10,5.5))
    fig_bee.patch.set_facecolor("#000d0d")
    ax.set_facecolor("#000d0d")

    shap.summary_plot(sv, X_samp, feature_names=feat_names,
                      max_display=14, show=False, plot_size=None,
                      color_bar=True, plot_type="dot",
                      color=plt.cm.RdYlGn)
    for spine in ax.spines.values():
        spine.set_edgecolor("#00ff9f")
        spine.set_linewidth(0.5)
    ax.tick_params(colors="#00c8ff", labelsize=9)
    ax.xaxis.label.set_color("#00c8ff")
    ax.yaxis.label.set_color("#00c8ff")
    ax.set_title("SHAP BEESWARM — GLOBAL FEATURE IMPACT",
                 color="#00ff9f", pad=12,
                 fontfamily="monospace", fontsize=11)
    ax.set_facecolor("#000d0d")
    plt.tight_layout()
    st.pyplot(fig_bee)
    plt.close(fig_bee)

    # Per-transaction waterfall
    sec("WATERFALL DECODE :: SINGLE TRANSACTION DISSECTION")
    tx_idx = st.slider("SELECT TRANSACTION INDEX", 0, len(X_samp)-1, 0)
    sv_row = sv[tx_idx]
    shap_s = pd.Series(sv_row, index=feat_names)
    top12  = shap_s.abs().nlargest(12).index
    shap_t = shap_s[top12]
    colors = ["#ff003c" if v>0 else "#00ff9f" for v in shap_t.values]

    x_vals   = [base_val]
    measures = ["relative"]
    text_vals= [f"BASE: {base_val:.3f}"]
    for feat in top12:
        x_vals.append(float(shap_s[feat]))
        measures.append("relative")
        text_vals.append(f"{feat}: {float(shap_s[feat]):+.4f}")
    x_vals.append(base_val+float(shap_s.sum()))
    measures.append("total")
    text_vals.append(f"SCORE: {base_val+float(shap_s.sum()):.3f}")

    fig_wf = go.Figure(go.Waterfall(
        orientation="v", measure=measures, y=x_vals,
        text=text_vals, textposition="outside",
        connector=dict(line=dict(color="rgba(0,255,159,.3)")),
        decreasing=dict(marker=dict(color="#00ff9f")),
        increasing=dict(marker=dict(color="#ff003c")),
        totals=dict(marker=dict(color="#00c8ff")),
        hovertemplate="%{text}<extra></extra>",
    ))
    fig_wf.update_layout(**PLOTLY_C, height=440,
        xaxis=dict(tickvals=list(range(len(x_vals))),
                   ticktext=["BASE"]+list(top12)+["FINAL"],
                   tickangle=-40, gridcolor="rgba(0,255,159,.08)"),
        yaxis=dict(title="SHAP CONTRIBUTION",gridcolor="rgba(0,255,159,.1)"),
        title=dict(text=f"// TRANSACTION #{tx_idx} — THREAT DECOMPOSITION",
                   font=dict(color="#00c8ff",size=12,family="Orbitron,monospace")),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    actual_lbl = "⚠ THREAT [FRAUD]" if int(y_s.iloc[tx_idx])==1 else "✓ CLEAR [LEGITIMATE]"
    color_lbl  = "#ff003c" if int(y_s.iloc[tx_idx])==1 else "#00ff9f"
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:.82rem;
                padding:10px 16px;background:rgba(0,255,159,.04);
                border:1px solid rgba(0,255,159,.25);border-radius:4px;
                color:{color_lbl};text-shadow:0 0 6px {color_lbl};">
        [ GROUND TRUTH — TX #{tx_idx} ] : {actual_lbl}
    </div>""", unsafe_allow_html=True)

else:
    st.warning("DATA FEED UNAVAILABLE. Run: python -m src.model_training")

st.markdown("---")
st.markdown("""<div style="text-align:center;font-family:'Share Tech Mono',monospace;
font-size:.62rem;color:rgba(0,255,159,.25);padding:8px 0;">
FRAUDSENTINEL · XAI_MODULE · GROUP_01 · 2026</div>""", unsafe_allow_html=True)
