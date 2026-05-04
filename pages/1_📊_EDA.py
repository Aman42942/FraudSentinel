"""
pages/1_📊_EDA.py  —  CYBER EDITION
FraudSentinel Capstone · Group 1
"""
import os, sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CSV_PATH = os.path.join(ROOT, "data", "creditcard_synthetic.csv")

st.set_page_config(page_title="EDA · FraudSentinel CYBER", page_icon="📊", layout="wide")

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
</style>""", unsafe_allow_html=True)

PLOTLY_C = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,13,13,0.9)",
                font=dict(family="Share Tech Mono,monospace", color="#00ff9f", size=11),
                margin=dict(l=20,r=20,t=44,b=20))

def sec(t):
    st.markdown(f'<div class="cyber-sec"><h2>// {t}</h2></div>', unsafe_allow_html=True)

st.markdown("""
<div style="padding:28px 0 16px;border-bottom:1px solid rgba(0,255,159,.2);">
<span style="font-family:'Orbitron',monospace;font-size:1.8rem;font-weight:900;
             color:#00ff9f;text-shadow:0 0 15px #00ff9f;">📊 DATA RECON</span>
<span style="font-family:'Share Tech Mono',monospace;font-size:.8rem;color:#00c8ff;
             margin-left:14px;text-shadow:0 0 8px #00c8ff;">
EXPLORATORY ANALYSIS MODULE</span></div>""", unsafe_allow_html=True)

@st.cache_data
def load_df():
    return pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else None

df = load_df()
if df is None:
    st.error("Dataset not found. Run: python -m src.model_training"); st.stop()

sec("DATASET OVERVIEW")
c1,c2,c3,c4 = st.columns(4)
c1.metric("TOTAL RECORDS",  f"{len(df):,}")
c2.metric("CLEAR SIGNALS",  f"{int((df['Class']==0).sum()):,}")
c3.metric("THREATS",        f"{int((df['Class']==1).sum()):,}")
c4.metric("FRAUD RATE",     f"{df['Class'].mean()*100:.3f}%")

with st.expander("[ RAW DATA FEED — FIRST 200 RECORDS ]"):
    st.dataframe(df.head(200), use_container_width=True, height=300)

sec("CLASS SIGNAL DISTRIBUTION")
cc = df["Class"].value_counts().reset_index()
cc.columns = ["Class","Count"]
cc["Label"] = cc["Class"].map({0:"LEGITIMATE",1:"FRAUD"})

fig_d = go.Figure(go.Pie(
    labels=cc["Label"], values=cc["Count"], hole=0.65,
    marker=dict(colors=["#00ff9f","#ff003c"],
                line=dict(color="#000d0d",width=3)),
    textinfo="label+percent",
    textfont=dict(size=12,color="#000d0d",family="Share Tech Mono"),
    hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
))
fig_d.update_layout(**PLOTLY_C, height=360,
    annotations=[dict(text="SIGNAL<br>SPLIT",x=0.5,y=0.5,showarrow=False,
                      font=dict(size=12,color="#00c8ff",family="Orbitron"))])
st.plotly_chart(fig_d, use_container_width=True)

sec("TRANSACTION AMOUNT ANALYSIS")
fig_a = go.Figure()
for cls,color,label in [(0,"#00ff9f","LEGITIMATE"),(1,"#ff003c","THREAT")]:
    fig_a.add_trace(go.Histogram(
        x=df.loc[df["Class"]==cls,"Amount"].clip(upper=2000),
        name=label, marker_color=color, opacity=0.7, nbinsx=60,
        hovertemplate=f"<b>{label}</b><br>Amount: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
    ))
fig_a.update_layout(**PLOTLY_C, height=360, barmode="overlay",
    xaxis=dict(title="AMOUNT (USD, clipped @2000)",gridcolor="rgba(0,255,159,.1)"),
    yaxis=dict(title="FREQUENCY",gridcolor="rgba(0,255,159,.08)"),
    legend=dict(bgcolor="rgba(0,0,0,.5)",bordercolor="rgba(0,255,159,.3)",borderwidth=1))
st.plotly_chart(fig_a, use_container_width=True)

sec("PCA FEATURE SIGNATURE ANALYSIS")
top_f = ["V1","V2","V3","V4","V10","V12","V14","V17"]
fig_b = go.Figure()
for cls,label,color in [(0,"LEGITIMATE","#00ff9f"),(1,"THREAT","#ff003c")]:
    sub = df[df["Class"]==cls]
    for feat in top_f:
        fig_b.add_trace(go.Box(
            y=sub[feat], name=feat, legendgroup=label,
            legendgrouptitle=dict(text=label) if feat==top_f[0] else None,
            marker_color=color, line_color=color, boxmean=True, opacity=0.75,
            hovertemplate=f"<b>{feat} ({label})</b><br>%{{y:.3f}}<extra></extra>",
        ))
fig_b.update_layout(**PLOTLY_C, height=420, boxmode="group",
    xaxis=dict(title="PCA COMPONENT",gridcolor="rgba(0,255,159,.08)"),
    yaxis=dict(title="VALUE",gridcolor="rgba(0,255,159,.1)"),
    legend=dict(bgcolor="rgba(0,0,0,.5)",bordercolor="rgba(0,255,159,.3)"))
st.plotly_chart(fig_b, use_container_width=True)

sec("CORRELATION MATRIX SCAN")
pca_c = [c for c in df.columns if c.startswith("V")][:15]
corr  = df[pca_c+["Amount","Class"]].corr()
fig_h = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
    colorscale=[[0,"#ff003c"],[0.5,"#000d0d"],[1,"#00ff9f"]], zmid=0,
    showscale=True, colorbar=dict(tickfont=dict(color="#00ff9f")),
    hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
))
fig_h.update_layout(**PLOTLY_C, height=500,
    xaxis=dict(tickangle=-45,gridcolor="rgba(0,255,159,.06)"),
    yaxis=dict(autorange="reversed",gridcolor="rgba(0,255,159,.06)"))
st.plotly_chart(fig_h, use_container_width=True)

st.markdown("---")
st.markdown("""<div style="text-align:center;font-family:'Share Tech Mono',monospace;
font-size:.62rem;color:rgba(0,255,159,.25);padding:8px 0;">
FRAUDSENTINEL · DATA_RECON_MODULE · GROUP_01 · 2026</div>""", unsafe_allow_html=True)
