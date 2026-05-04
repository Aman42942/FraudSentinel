"""
src/data_processing.py
======================
Data loading, feature scaling, train-test splitting, and SMOTE oversampling
for the Financial Fraud Detection capstone project (Group 1).

Mathematical note on SMOTE application order
---------------------------------------------
SMOTE must be applied STRICTLY to the training set AFTER the train-test split.
If we oversample before splitting, synthetic minority samples derived from real
training examples will "leak" neighbourhood information into the test fold,
producing an artificially inflated recall and PR-AUC — a form of data leakage
that makes the published metrics unrepresentative of real-world performance.
By keeping the held-out test set entirely untouched, the evaluation remains
honest and unbiased.

RobustScaler is chosen over StandardScaler because financial transaction amounts
follow heavy-tailed distributions with extreme outliers (e.g., a single high-
value wire transfer). RobustScaler subtracts the median and divides by the
interquartile range (IQR), so those extreme values do not distort the scale of
the majority of transactions, as a mean-based scaler would.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
CSV_PATH     = os.path.join(DATA_DIR, "creditcard_synthetic.csv")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Synthetic dataset generator ───────────────────────────────────────────────
def generate_synthetic_dataset(n_rows: int = 10_000, fraud_rate: float = 0.0172,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset that mirrors the structure of the
    European Credit Card Fraud dataset (Kaggle / ULB).

    The 28 PCA-transformed features (V1-V28) are modelled as correlated Gaussian
    noise for legitimate transactions, and as shifted Gaussian distributions for
    fraudulent ones — reproducing the statistical separation visible in the real
    dataset.  Time and Amount are drawn from empirically plausible distributions.

    Parameters
    ----------
    n_rows       : Total number of transactions to generate.
    fraud_rate   : Proportion of fraudulent transactions (default ≈ 0.17 %).
    random_state : Seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns [Time, V1..V28, Amount, Class].
    """
    rng       = np.random.default_rng(random_state)
    n_fraud   = max(1, int(n_rows * fraud_rate))
    n_legit   = n_rows - n_fraud

    # PCA feature means that separate fraud from legitimate transactions
    # (estimated from published EDA of the real dataset)
    fraud_mean_shift = np.array([
        -3.0,  3.5, -4.0,  4.2, -2.8,  3.1, -2.5,  1.8,
        -2.0,  2.3, -1.5,  1.2, -1.8,  1.0, -0.5,  0.8,
        -0.3,  0.5, -0.2,  0.4, -0.1,  0.3, -0.1,  0.2,
        -0.05, 0.1, -0.05, 0.05
    ])

    # Legitimate transactions: mean ≈ 0, unit variance
    V_legit = rng.standard_normal((n_legit, 28))

    # Fraudulent transactions: shifted means + slightly tighter variance
    V_fraud = (rng.standard_normal((n_fraud, 28)) * 0.8 + fraud_mean_shift)

    # Time feature: seconds elapsed over 48 h
    time_legit = rng.uniform(0, 172_800, n_legit)
    time_fraud = rng.uniform(0, 172_800, n_fraud)

    # Amount: log-normal for legitimate (μ=3.5, σ=1.2), lower for fraud
    amount_legit = rng.lognormal(mean=3.5, sigma=1.2, size=n_legit).clip(0.5, 25_000)
    amount_fraud = rng.lognormal(mean=2.0, sigma=1.0, size=n_fraud).clip(0.5, 2_500)

    col_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

    legit_df = pd.DataFrame(
        np.column_stack([time_legit, V_legit, amount_legit, np.zeros(n_legit)]),
        columns=col_names,
    )
    fraud_df = pd.DataFrame(
        np.column_stack([time_fraud, V_fraud, amount_fraud, np.ones(n_fraud)]),
        columns=col_names,
    )

    df = pd.concat([legit_df, fraud_df], ignore_index=True).sample(
        frac=1, random_state=random_state
    )
    df["Class"] = df["Class"].astype(int)
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────
def load_and_preprocess(csv_path: str = CSV_PATH,
                        test_size: float = 0.20,
                        random_state: int = 42,
                        smote_k_neighbors: int = 5):
    """
    Full preprocessing pipeline:
      1. Load (or generate) the dataset.
      2. Apply RobustScaler to Time and Amount.
      3. Stratified 80/20 train-test split.
      4. Apply SMOTE **only** to the training split.
      5. Persist the fitted scaler to models/.

    Parameters
    ----------
    csv_path        : Path to the CSV file.
    test_size       : Fraction of data reserved for testing.
    random_state    : Reproducibility seed.
    smote_k_neighbors : k for SMOTE's nearest-neighbour search.

    Returns
    -------
    Tuple: (X_train_res, y_train_res, X_test, y_test, scaler, feature_names)
    """
    # ── 1. Load / generate data ────────────────────────────────────────────
    if os.path.exists(csv_path):
        print(f"[data_processing] Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("[data_processing] CSV not found — generating synthetic dataset …")
        df = generate_synthetic_dataset()
        df.to_csv(csv_path, index=False)
        print(f"[data_processing] Synthetic dataset saved to: {csv_path}")

    print(f"[data_processing] Dataset shape: {df.shape}")
    print(f"[data_processing] Class distribution:\n{df['Class'].value_counts()}\n")

    # ── 2. Feature/target separation ──────────────────────────────────────
    target       = "Class"
    features     = [c for c in df.columns if c != target]
    X            = df[features].copy()
    y            = df[target].copy()

    # ── 3. RobustScaler on Time and Amount ────────────────────────────────
    scaler       = RobustScaler()
    scale_cols   = [c for c in ["Time", "Amount"] if c in X.columns]
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    # ── 4. Stratified train-test split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[data_processing] Train size: {X_train.shape[0]}  |  "
          f"Test size: {X_test.shape[0]}")

    # ── 5. SMOTE on training set only ─────────────────────────────────────
    print("[data_processing] Applying SMOTE to training set …")
    smote = SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[data_processing] Post-SMOTE training class distribution: "
          f"{pd.Series(y_train_res).value_counts().to_dict()}")

    # ── 6. Persist scaler ─────────────────────────────────────────────────
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[data_processing] Scaler saved -> {scaler_path}")

    return X_train_res, y_train_res, X_test, y_test, scaler, features


if __name__ == "__main__":
    load_and_preprocess()
