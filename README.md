# 🛡️ FraudSentinel — Detecting Financial Fraud Using Machine Learning

> **Group 1 Capstone · Advanced Machine Learning · 2026**  
> Abhishek Kumar Saroj · Aman Kumar · Amit · Ankit

---

## 📌 Project Overview

Financial fraud costs the global economy hundreds of billions of dollars annually. Static, rule-based detection systems struggle to adapt as adversarial actors constantly probe for gaps. This capstone project delivers a **production-grade, end-to-end machine learning pipeline** that tackles the problem from raw data ingestion all the way through to an interactive, publicly accessible web dashboard — with zero infrastructure cost.

The core insight driving our architecture: fraud is not a rare coincidence, it is a *structural anomaly*. If the learning algorithm can be shown enough diverse examples of what fraud *looks like* in feature space, it can generalise those patterns to catch novel attacks in real time.

---

## 🏗️ Project Structure

```
final project/
├── data/
│   └── creditcard_synthetic.csv      ← auto-generated 10 k-row dataset
├── src/
│   ├── __init__.py
│   ├── data_processing.py            ← RobustScaler · train-test split · SMOTE
│   └── model_training.py             ← XGBoost · evaluation · serialisation
├── models/
│   ├── xgb_model.joblib              ← trained classifier
│   ├── scaler.joblib                 ← fitted RobustScaler
│   └── metrics.joblib                ← saved evaluation metrics
├── pages/
│   ├── 1_📊_EDA.py                   ← Exploratory Data Analysis
│   ├── 2_🤖_Model_Training.py        ← Metrics, PR-AUC curve, feature importance
│   └── 3_🔍_XAI_Explainability.py    ← SHAP global & per-transaction analysis
├── app.py                            ← Main dashboard (KPIs, live prediction)
├── requirements.txt
└── README.md
```

---

## 📐 Mathematical Foundations

### Why RobustScaler?

Transaction amounts in financial datasets follow a heavy-tailed distribution — a single large wire transfer can be orders of magnitude larger than a typical grocery purchase. A standard Z-score scaler subtracts the **mean** and divides by **standard deviation**, both of which are sensitive to those extreme values. The RobustScaler instead applies:

$$x_{\text{scaled}} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$

Because the median and interquartile range are inherently resistant to outliers, the transformed features preserve the relative structure of the bulk of transactions without letting extreme-value transactions warp the scale.

### Why SMOTE — and why *only* on training data?

The European Credit Card Fraud dataset exhibits approximately **0.172% fraud** — roughly 1 fraudulent transaction per 582 legitimate ones. Feeding this raw ratio directly into a classifier creates a pathological incentive: the model achieves 99.83% accuracy by *never* predicting fraud. That is not a detection system; it is a false sense of security.

SMOTE (Synthetic Minority Over-sampling Technique) resolves this by *synthesising* new minority-class examples rather than merely duplicating existing ones:

$$x_{\text{new}} = x_i + \lambda \cdot (\hat{x}_i - x_i), \quad \lambda \sim \mathcal{U}(0,1)$$

where $x_i$ is a real fraud transaction and $\hat{x}_i$ is one of its $k$ nearest minority neighbours in feature space. The interpolated point $x_{\text{new}}$ lies on the line segment between them — mathematically plausible, never a direct copy.

**Critically, SMOTE is applied exclusively to the training set**, after the stratified 80/20 split. Applying it before the split would cause synthetic points derived from real training samples to appear in the test fold, artificially inflating recall and PR-AUC — a textbook case of **data leakage** that renders all validation metrics meaningless.

### Why XGBoost?

XGBoost minimises a regularised objective over an ensemble of $K$ trees:

$$\text{Obj}(\Theta) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

The regularisation term $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda \|w\|^2$ penalises the number of leaves $T$ and the magnitude of leaf weights $w$, directly preventing the model from memorising the SMOTE-augmented training set. Sequential boosting then focuses each new tree on the residual errors of its predecessors, achieving superior recall on hard-to-classify edge cases.

### Why PR-AUC instead of ROC-AUC?

In a dataset where 99.83% of transactions are legitimate, a trivial all-negative classifier still achieves an ROC-AUC above 0.90. The massive True Negative count collapses the False Positive Rate toward zero regardless of how poorly the model performs on fraud. The **Precision-Recall curve** is immune to this: it ignores True Negatives entirely, examining only how many fraud alerts were correct (Precision) and how much of the actual fraud was caught (Recall). PR-AUC is therefore the gold standard for imbalanced binary classification in the financial domain.

---

## ⚙️ Installation & Local Setup

### 1. Clone or download the repository

```bash
git clone https://github.com/<your-org>/fraudsentinel.git
cd fraudsentinel
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (generates dataset + model artefacts)

```bash
python -m src.model_training
```

This single command:
- Auto-generates `data/creditcard_synthetic.csv` (10 000 rows, ~0.17% fraud)
- Applies RobustScaler and SMOTE
- Trains an XGBClassifier and prints the full evaluation report
- Saves `models/xgb_model.joblib`, `models/scaler.joblib`, `models/metrics.joblib`

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🚀 Zero-Cost Deployment — Streamlit Community Cloud

Streamlit Community Cloud provides **free, unlimited public hosting** with automatic CI/CD triggered by GitHub pushes.

### Step-by-step deployment

1. **Push your project to GitHub**
   ```bash
   git init
   git remote add origin https://github.com/<your-username>/fraudsentinel.git
   git add .
   git commit -m "feat: initial capstone project"
   git push -u origin main
   ```

2. **Sign in to Streamlit Community Cloud**  
   Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.

3. **Create a new app**  
   - Click **"New app"**
   - Repository: `<your-username>/fraudsentinel`
   - Branch: `main`
   - Main file path: `app.py`
   - Click **"Deploy!"**

4. **Wait for the build**  
   Streamlit Cloud reads `requirements.txt`, installs all packages, and launches the server. Your app will be live at:
   ```
   https://<your-username>-fraudsentinel-app-XXXXX.streamlit.app
   ```

5. **Automatic re-deployment**  
   Every subsequent `git push` to `main` triggers an automatic rebuild. No manual steps required.

> **Note:** The first run will train the model inside the cloud container because `models/` is empty in a fresh clone. To avoid this, commit the pre-trained `.joblib` files to Git (they are ~few MB each). Alternatively, add a `setup.sh` script that runs `python -m src.model_training` before the Streamlit server starts.

---

## 🌿 GitHub Branching Strategy — Equal Contribution Tracking

To satisfy the academic requirement for **provable, equal contribution** among all four group members, the project follows **GitHub Flow**:

### Branch ownership

| Member | Branch | Issue |
|---|---|---|
| Amit | `feature/data-preprocessing` | #1 — Data Cleaning & EDA |
| Ankit | `feature/smote-scaling` | #2 — SMOTE Implementation & Scaling |
| Aman Kumar | `feature/xgboost-shap` | #3 — XGBoost Training & SHAP Integration |
| Abhishek Kumar Saroj | `feature/streamlit-ui` | #4 — Streamlit Dashboard & Deployment |

### Workflow

```bash
# 1. Create your feature branch
git checkout -b feature/your-feature-name

# 2. Make commits with descriptive messages
git add .
git commit -m "feat: implement RobustScaler on Time and Amount features"

# 3. Push to GitHub
git push origin feature/your-feature-name

# 4. Open a Pull Request to main
# 5. Another group member reviews, approves, and merges
```

**Why this matters to evaluators:** The GitHub **Insights → Contributors** tab provides a timestamped, per-member breakdown of commits, lines added, and pull requests. This is irrefutable empirical evidence of individual workload distribution.

---

## 📊 Evaluation Metrics Summary

| Metric | Purpose |
|---|---|
| **PR-AUC** | Primary metric — threshold-independent, imbalance-robust |
| **Recall** | Fraction of actual fraud successfully intercepted |
| **Precision** | Fraction of fraud alerts that were genuinely fraudulent |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **MCC** | Matthews Correlation Coefficient — reliable for imbalanced data |
| **G-Mean** | Geometric mean of sensitivity and specificity |
| **ROC-AUC** | Secondary metric (can be misleading under high imbalance) |

---

## 🤖 AI Acknowledgement Appendix

This project was developed with assistance from **Google Antigravity** (agentic AI coding assistant). The AI tools were used to:

- Generate the initial scaffolding, file structure, and boilerplate code
- Suggest mathematical formulations for SMOTE and XGBoost regularisation
- Draft documentation and README sections

**All AI-generated outputs were:**
- Reviewed line-by-line by each contributing member
- Refactored to align with lecture terminology and university-specific notation
- Augmented with inline comments written in the students' own words
- Validated by running the training pipeline end-to-end and inspecting outputs

The underlying algorithms, architectural decisions, and evaluation strategy reflect the independent academic judgement of Group 1, not a copy-paste of AI output.

---

## 📜 License

This project is submitted for academic assessment purposes. All code is original work of Group 1, produced for the Advanced Machine Learning course. External libraries are used under their respective open-source licences (MIT, BSD-3-Clause).

---

*FraudSentinel v1.0 · Advanced Machine Learning · 2026*
