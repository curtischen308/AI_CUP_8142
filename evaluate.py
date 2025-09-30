# evaluate.py
import pandas as pd
from sklearn.metrics import f1_score

def evaluate_local_f1(submission_path="outputs/submission.csv", alerts_path="dataset/acct_alert.csv"):
    sub = pd.read_csv(submission_path)
    alerts = pd.read_csv(alerts_path)

    # 欄位容錯
    if "acct" not in sub.columns:
        for c in ["acct_id","account","account_id","id"]:
            if c in sub.columns:
                sub = sub.rename(columns={c:"acct"})
                break
    if "acct_id" in alerts.columns:
        alerts = alerts.rename(columns={"acct_id":"acct"})

    alerts["label"] = 1
    df = sub.merge(alerts[["acct","label"]], on="acct", how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    if "pred" not in df.columns:
        raise ValueError("submission 檔缺少欄位 'pred'。")
    f1 = f1_score(df["label"], df["pred"])
    print(f"📊 本地驗證 F1 = {f1:.4f}（僅供參考，非官方 leaderboard 分數）")
    return f1
