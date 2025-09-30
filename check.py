import pandas as pd
from pathlib import Path

# 讀取提交檔
sub_path = Path("outputs/submission_final.csv")
sub = pd.read_csv(sub_path)

print("✅ submission_final.csv 已讀取")
print(f"共有 {len(sub)} 筆")
print("前 5 筆：")
print(sub.head())

# 檢查欄位
assert set(sub.columns) == {"acct", "label"}, "⚠️ submission_final.csv 欄位必須是 acct,label"

# 檢查 label 分布
label_counts = sub["label"].value_counts()
print("\n📊 Label 分布：")
print(label_counts)

# 如果你用 dataset/acct_predict.csv，檢查行數一致
pred_path = Path("dataset/acct_predict.csv")
if pred_path.exists():
    pred = pd.read_csv(pred_path)
    print(f"\n測試檔有 {len(pred)} 筆")
    if len(pred) != len(sub):
        print("⚠️ 筆數不一致！")
    else:
        print("✅ 筆數一致")
