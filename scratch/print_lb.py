import joblib
lb = joblib.load('models/leaderboard.joblib')
for r in lb:
    print(f"#{r['rank']} {r['model']}: PR-AUC={r['pr_auc']}, ROC={r['roc_auc']}, F1={r['f1_score']}, Prec={r['precision']}, Rec={r['recall']}, MCC={r['mcc']}, Lat={r['latency_ms']}ms, Train={r['train_time']}s")

import pandas as pd
df = pd.read_csv('data/creditcard_synthetic.csv')
print(f"\nDataset: {df.shape}")
print(df['Class'].value_counts())
print(f"Fraud rate: {df['Class'].mean()*100:.4f}%")
print(f"Columns: {list(df.columns)}")
