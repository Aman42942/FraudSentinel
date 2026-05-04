import joblib
try:
    data = joblib.load('models/comparison.joblib')
    print("| Model | PR-AUC | ROC-AUC | F1-Score | Precision | Recall |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for m, v in data.items():
        pr = v.get('pr_auc', 0)
        roc = v.get('roc_auc', 0)
        f1 = v.get('f1_score', 0)
        prec = v.get('precision', 0)
        rec = v.get('recall', 0)
        print(f"| {m} | {pr:.4f} | {roc:.4f} | {f1:.4f} | {prec:.4f} | {rec:.4f} |")
except Exception as e:
    print(f"Error: {e}")
