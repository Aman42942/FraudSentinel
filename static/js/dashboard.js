/* ============================================================
   dashboard.js — FraudSentinel · All Dynamic Features
   ============================================================ */

"use strict";

/* ── Config ────────────────────────────────────────────────── */
const FEED_INTERVAL   = 1800;   // ms between live transactions
const SHAP_DEBOUNCE   = 600;    // ms debounce on slider change
const STATS_INTERVAL  = 5000;   // ms between stat refresh
const FEED_MAX        = 20;     // max entries in feed

/* ── Audio Alert ───────────────────────────────────────────── */
const AudioCtx = window.AudioContext || window.webkitAudioContext;
let   audioCtx = null;

function playAlert(freq=440, dur=0.3, type="sine") {
  try {
    if (!audioCtx) audioCtx = new AudioCtx();
    const osc  = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.type = type; osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
    osc.start(); osc.stop(audioCtx.currentTime + dur);
  } catch(e){}
}

function playFraudAlert() {
  playAlert(880, 0.15, "square");
  setTimeout(()=>playAlert(660, 0.15, "square"), 160);
  setTimeout(()=>playAlert(440, 0.25, "square"), 320);
}

/* ── Toast Notifications ───────────────────────────────────── */
function showToast(msg, type="info", duration=4000) {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${type==="fraud"?"⚠":type==="success"?"✓":"ℹ"}</span>
    <span class="toast-msg">${msg}</span>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>`;
  container.appendChild(toast);
  requestAnimationFrame(()=> toast.classList.add("show"));
  setTimeout(()=>{ toast.classList.remove("show");
                   setTimeout(()=>toast.remove(), 300); }, duration);
}

/* ── Counter Animation ─────────────────────────────────────── */
function animateCounter(el, target, duration=1200, prefix="", suffix="") {
  const start    = 0;
  const startTime= performance.now();
  function step(now) {
    const elapsed = now - startTime;
    const progress= Math.min(elapsed/duration, 1);
    const eased   = 1 - Math.pow(1-progress, 3);
    const current = Math.floor(start + (target - start) * eased);
    el.textContent = prefix + current.toLocaleString() + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function initCounters() {
  document.querySelectorAll("[data-counter]").forEach(el => {
    const raw    = parseFloat(el.dataset.counter);
    const prefix = el.dataset.prefix || "";
    const suffix = el.dataset.suffix || "";
    const isFloat= el.dataset.float === "true";
    if (isFloat) {
      // For decimal values like 0.998
      const dur = 1200;
      const start = performance.now();
      function step(now) {
        const t = Math.min((now-start)/dur, 1);
        const e = 1-Math.pow(1-t,3);
        el.textContent = prefix + (raw*e).toFixed(3) + suffix;
        if(t<1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    } else {
      animateCounter(el, raw, 1200, prefix, suffix);
    }
  });
}

/* ── Live Transaction Feed ─────────────────────────────────── */
let feedRunning  = false;
let feedInterval = null;
let feedCount    = { total:0, fraud:0 };

function startLiveFeed() {
  if (feedRunning) return;
  feedRunning = true;
  const btn = document.getElementById("feed-toggle");
  if (btn) { btn.textContent="⏹ STOP FEED"; btn.classList.add("active"); }
  feedInterval = setInterval(fetchOneTx, FEED_INTERVAL);
  fetchOneTx(); // immediate first call
}

function stopLiveFeed() {
  feedRunning = false;
  clearInterval(feedInterval);
  const btn = document.getElementById("feed-toggle");
  if (btn) { btn.textContent="▶ START FEED"; btn.classList.remove("active"); }
}

function toggleFeed() {
  feedRunning ? stopLiveFeed() : startLiveFeed();
}

async function fetchOneTx() {
  try {
    const res  = await fetch("/api/live-tx");
    const data = await res.json();
    addFeedEntry(data);
    updateFeedStats(data);
  } catch(e) { console.error("Feed error:", e); }
}

function addFeedEntry(tx) {
  const list = document.getElementById("live-feed-list");
  if (!list) return;

  const isFraud = tx.pred === 1;
  const row = document.createElement("div");
  row.className = `feed-row ${isFraud ? "feed-fraud" : "feed-legit"} feed-new`;
  row.innerHTML = `
    <span class="feed-id">${tx.id}</span>
    <span class="feed-type">${tx.type}</span>
    <span class="feed-amt">$${parseFloat(tx.amount).toFixed(2)}</span>
    <span class="feed-score" style="color:${isFraud?"#ff003c":"#00ff9f"}">
      ${(tx.prob*100).toFixed(1)}%
    </span>
    <span class="feed-status ${isFraud?"status-threat":"status-clear"}">
      ${isFraud?"⚠ THREAT":"✓ CLEAR"}
    </span>
    <span class="feed-time">${tx.time}</span>`;
  list.prepend(row);
  setTimeout(()=>row.classList.remove("feed-new"), 50);

  // Remove oldest if > max
  while (list.children.length > FEED_MAX) list.lastChild.remove();

  // Fraud alert
  if (isFraud) {
    playFraudAlert();
    showToast(`⚠ FRAUD: ${tx.id} — $${parseFloat(tx.amount).toFixed(2)} — Score: ${(tx.prob*100).toFixed(1)}%`, "fraud", 5000);
    document.getElementById("live-feed-list")?.classList.add("flash-red");
    setTimeout(()=>document.getElementById("live-feed-list")?.classList.remove("flash-red"), 500);
  }
}

function updateFeedStats(tx) {
  feedCount.total++;
  if (tx.pred===1) feedCount.fraud++;

  const elTotal = document.getElementById("feed-total");
  const elFraud = document.getElementById("feed-fraud");
  const elRate  = document.getElementById("feed-rate");
  if (elTotal) elTotal.textContent = feedCount.total.toLocaleString();
  if (elFraud) elFraud.textContent = feedCount.fraud.toLocaleString();
  if (elRate) {
    const rate = (feedCount.fraud/feedCount.total*100).toFixed(1);
    elRate.textContent = rate + "%";
    elRate.style.color = parseFloat(rate) > 5 ? "#ff003c" : "#00ff9f";
  }
}

/* ── Real-time SHAP (debounced) ────────────────────────────── */
let shapTimer = null;

function scheduleShap() {
  clearTimeout(shapTimer);
  shapTimer = setTimeout(runAutoShap, SHAP_DEBOUNCE);
}

async function runAutoShap() {
  const indicator = document.getElementById("shap-loading");
  if (indicator) indicator.style.display="inline";

  const payload = buildPayload();
  try {
    const [predRes, shapRes] = await Promise.all([
      fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}),
      fetch("/api/shap",   {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}),
    ]);
    const pred = await predRes.json();
    const shap = await shapRes.json();
    renderPredResult(pred);
    renderShapBar(shap.shap_values);
    renderWaterfall(shap.shap_values, shap.base_value);
    updateRiskGauge(pred.probability);
  } catch(e) { console.error("SHAP error:", e); }

  if (indicator) indicator.style.display="none";
}

function buildPayload(threshold=0.5) {
  const FEAT = window.FEAT_NAMES || [];
  const payload = {threshold};
  FEAT.forEach(f => {
    const el = document.getElementById("x-"+f) || document.getElementById("f-"+f);
    payload[f] = el ? parseFloat(el.value) : 0.0;
  });
  return payload;
}

function renderPredResult(pred) {
  const box   = document.getElementById("result-box") || document.getElementById("shap-pred-box");
  const label = document.getElementById("result-label") || document.getElementById("shap-pred-label");
  const score = document.getElementById("result-score") || document.getElementById("shap-pred-score");
  if (!box) return;
  const isFraud = pred.prediction===1;
  box.className = "result-box show " + (isFraud?"result-fraud":"result-legit");
  if (label) label.textContent = isFraud?"⚠ FRAUDULENT TRANSACTION":"✓ TRANSACTION AUTHORIZED";
  if (score) score.textContent = `THREAT_SCORE: ${(pred.probability*100).toFixed(2)}%`;
}

function renderShapBar(shapVals) {
  const el = document.getElementById("shap-bar-chart");
  if (!el || typeof Plotly==="undefined") return;
  const entries = Object.entries(shapVals).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  const feats   = entries.map(e=>e[0]);
  const vals    = entries.map(e=>e[1]);
  const colors  = vals.map(v=>v>0?"#ff003c":"#00ff9f");
  Plotly.react(el,[{
    type:"bar",orientation:"h",x:vals,y:feats,
    marker:{color:colors},
    hovertemplate:"<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
  }],{
    paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,13,13,.95)",
    font:{family:"Share Tech Mono,monospace",color:"#00ff9f",size:11},
    margin:{l:10,r:10,t:10,b:30},height:320,
    xaxis:{title:"SHAP Impact",gridcolor:"rgba(0,255,159,.08)",
           zerolinecolor:"rgba(0,255,159,.3)",color:"#00c8ff"},
    yaxis:{gridcolor:"rgba(0,255,159,.05)",color:"#00c8ff"},
  },{responsive:true,displayModeBar:false});
  document.getElementById("shap-section") &&
    (document.getElementById("shap-section").style.display="block");
}

function renderWaterfall(shapVals, baseVal) {
  const el = document.getElementById("shap-waterfall-chart");
  if (!el || typeof Plotly==="undefined") return;
  const entries = Object.entries(shapVals).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  const feats   = entries.map(e=>e[0]);
  const vals    = entries.map(e=>e[1]);
  const measures= ["relative",...Array(vals.length).fill("relative"),"total"];
  const ys      = [baseVal,...vals, baseVal+vals.reduce((a,b)=>a+b,0)];
  const texts   = [`BASE: ${baseVal.toFixed(3)}`,
                   ...feats.map((f,i)=>`${f}: ${vals[i]>0?"+":""}${vals[i].toFixed(4)}`),
                   `FINAL: ${(baseVal+vals.reduce((a,b)=>a+b,0)).toFixed(3)}`];
  Plotly.react(el,[{
    type:"waterfall",orientation:"v",
    measure:measures, y:ys, text:texts, textposition:"outside",
    connector:{line:{color:"rgba(0,255,159,.2)"}},
    decreasing:{marker:{color:"#00ff9f"}},
    increasing:{marker:{color:"#ff003c"}},
    totals:{marker:{color:"#00c8ff"}},
    hovertemplate:"%{text}<extra></extra>",
  }],{
    paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,13,13,.95)",
    font:{family:"Share Tech Mono,monospace",color:"#00ff9f",size:11},
    margin:{l:10,r:10,t:10,b:30},height:360,
    xaxis:{tickvals:[...Array(ys.length).keys()],
           ticktext:["BASE",...feats,"FINAL"],
           tickangle:-40,gridcolor:"rgba(0,255,159,.06)",color:"#00c8ff"},
    yaxis:{title:"SHAP",gridcolor:"rgba(0,255,159,.08)",color:"#00c8ff"},
  },{responsive:true,displayModeBar:false});
}

function updateRiskGauge(prob) {
  const el = document.getElementById("risk-gauge");
  if (!el || typeof Plotly==="undefined") return;
  Plotly.react(el,[{
    type:"indicator", mode:"gauge+number",
    value:prob*100,
    number:{suffix:"%",font:{family:"Orbitron,monospace",color:prob>0.5?"#ff003c":"#00ff9f",size:26}},
    gauge:{
      axis:{range:[0,100],tickcolor:"#00c8ff",tickfont:{color:"#00c8ff",size:8}},
      bar:{color:prob>0.5?"#ff003c":"#00ff9f",thickness:0.25},
      bgcolor:"rgba(0,13,13,0.9)",borderwidth:1,
      bordercolor:"rgba(0,255,159,0.2)",
      steps:[{range:[0,50],color:"rgba(0,255,159,0.06)"},
             {range:[50,100],color:"rgba(255,0,60,0.08)"}],
    },
  }],{
    paper_bgcolor:"rgba(0,0,0,0)",
    font:{color:"#00ff9f"},
    height:180,margin:{l:20,r:20,t:10,b:10},
  },{responsive:true,displayModeBar:false});
}

/* ── Slider value display ──────────────────────────────────── */
function updateVal(name, val) {
  const el = document.getElementById("v-"+name)||document.getElementById("xv-"+name);
  if (el) el.textContent = parseFloat(val).toFixed(1);
}

/* ── Batch Upload ──────────────────────────────────────────── */
async function runBatchUpload() {
  const fileInput = document.getElementById("batch-file");
  const threshold = document.getElementById("batch-threshold")?.value || 0.5;
  if (!fileInput?.files.length) {
    showToast("Please select a CSV file first.", "info"); return;
  }
  const btn = document.getElementById("batch-btn");
  btn.innerHTML='<span class="spinner"></span> PROCESSING...';
  btn.disabled=true;
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  fd.append("threshold", threshold);
  try {
    const res  = await fetch("/api/batch-predict",{method:"POST",body:fd});
    const data = await res.json();
    renderBatchResults(data);
    showToast(`Batch complete: ${data.fraud} threats in ${data.total} records.`,
              data.fraud>0?"fraud":"success");
  } catch(e){ showToast("Batch prediction failed.","info"); }
  btn.innerHTML="⚡ RUN BATCH SCAN"; btn.disabled=false;
}

function renderBatchResults(data) {
  const box = document.getElementById("batch-results");
  if (!box) return;
  box.style.display="block";
  const el = document.getElementById("batch-stats");
  if (el) el.innerHTML = `
    <div class="mini-stat"><span class="mini-val">${data.total.toLocaleString()}</span><span class="mini-lbl">TOTAL</span></div>
    <div class="mini-stat bad"><span class="mini-val" style="color:#ff003c">${data.fraud.toLocaleString()}</span><span class="mini-lbl">THREATS</span></div>
    <div class="mini-stat good"><span class="mini-val" style="color:#00ff9f">${data.legit.toLocaleString()}</span><span class="mini-lbl">CLEAR</span></div>
    <div class="mini-stat"><span class="mini-val" style="color:#f59e0b">${data.fraud_rate}%</span><span class="mini-lbl">FRAUD RATE</span></div>`;
  const tbody = document.getElementById("batch-table-body");
  if (tbody && data.rows) {
    tbody.innerHTML = data.rows.slice(0,50).map(r=>`
      <tr class="${r.PREDICTION==='FRAUD'?'row-fraud':''}">
        <td style="color:${r.PREDICTION==='FRAUD'?'#ff003c':'#00ff9f'}">${r.PREDICTION}</td>
        <td>${(parseFloat(r.THREAT_SCORE)*100).toFixed(1)}%</td>
        <td>$${parseFloat(r.Amount||0).toFixed(2)}</td>
        <td>${r.Time||'–'}</td>
      </tr>`).join("");
  }
}

/* ── Simulator page ────────────────────────────────────────── */
let simRunning=false, simInterval=null, simData=[];
let simFraud=0, simTotal=0;

function startSimulator() {
  if (simRunning) return;
  simRunning=true;
  document.getElementById("sim-btn").textContent="⏹ STOP SIMULATION";
  document.getElementById("sim-btn").classList.add("active");
  simInterval = setInterval(simTick, 600);
  simTick();
}

function stopSimulator() {
  simRunning=false;
  clearInterval(simInterval);
  document.getElementById("sim-btn").textContent="▶ START SIMULATION";
  document.getElementById("sim-btn").classList.remove("active");
}

function toggleSimulator() { simRunning ? stopSimulator() : startSimulator(); }

async function simTick() {
  try {
    const res  = await fetch("/api/live-tx");
    const data = await res.json();
    simTotal++;
    if (data.pred===1) simFraud++;
    simData.push({t:simTotal, fraud:simFraud/simTotal*100,
                  score:data.prob*100, pred:data.pred});
    if (simData.length>60) simData.shift();
    updateSimCharts();
    updateSimCounters();
    addSimLog(data);
  } catch(e){}
}

function updateSimCharts() {
  const el = document.getElementById("sim-chart");
  if (!el||typeof Plotly==="undefined") return;
  Plotly.react(el,[
    {x:simData.map(d=>d.t), y:simData.map(d=>d.fraud),
     type:"scatter",mode:"lines",name:"FRAUD RATE %",
     line:{color:"#ff003c",width:2},
     fill:"tozeroy",fillcolor:"rgba(255,0,60,0.07)"},
    {x:simData.map(d=>d.t), y:simData.map(d=>d.score),
     type:"scatter",mode:"lines",name:"THREAT SCORE %",
     line:{color:"#00c8ff",width:1.5},yaxis:"y2"},
  ],{
    paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,13,13,.95)",
    font:{family:"Share Tech Mono",color:"#00ff9f",size:10},
    margin:{l:40,r:40,t:10,b:30},height:300,
    xaxis:{title:"Transaction #",color:"#00c8ff",gridcolor:"rgba(0,255,159,.07)"},
    yaxis:{title:"Fraud Rate %",color:"#ff003c",gridcolor:"rgba(0,255,159,.07)"},
    yaxis2:{title:"Threat Score %",overlaying:"y",side:"right",color:"#00c8ff"},
    legend:{font:{color:"#00ff9f"},bgcolor:"rgba(0,0,0,.5)"},
    showlegend:true,
  },{responsive:true,displayModeBar:false});
}

function updateSimCounters() {
  const elT = document.getElementById("sim-total");
  const elF = document.getElementById("sim-fraud");
  const elR = document.getElementById("sim-rate");
  if (elT) elT.textContent=simTotal.toLocaleString();
  if (elF) elF.textContent=simFraud.toLocaleString();
  if (elR) {
    const r=(simFraud/Math.max(simTotal,1)*100).toFixed(1)+"%";
    elR.textContent=r;
    elR.style.color=simFraud/simTotal>0.1?"#ff003c":"#00ff9f";
  }
}

function addSimLog(tx) {
  const log = document.getElementById("sim-log");
  if (!log) return;
  const div=document.createElement("div");
  div.className=`sim-log-entry ${tx.pred===1?"log-fraud":"log-legit"}`;
  div.textContent=`[${tx.time}] ${tx.id} | $${tx.amount.toFixed(2)} | ${(tx.prob*100).toFixed(1)}% | ${tx.pred===1?"⚠ FRAUD":"✓ CLEAR"}`;
  log.prepend(div);
  while(log.children.length>25) log.lastChild.remove();
}

/* ── Export CSV ────────────────────────────────────────────── */
function exportCSV() {
  window.open("/api/export-csv","_blank");
}

/* ── Init ──────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", ()=>{
  initCounters();
  // Auto-start feed on dashboard
  if (document.getElementById("live-feed-list")) {
    setTimeout(startLiveFeed, 800);
  }
  // Auto-start SHAP on XAI page
  if (document.getElementById("shap-bar-chart")) {
    setTimeout(runAutoShap, 400);
  }
});
