XGBoost 0.0956175

以 XGBoost 建立帳戶層級的風險二元分類模型。流程分三段：

由交易明細產生帳號特徵（account_features.py）

訓練與推論（main.py，底層呼叫 model_xgb.py）

產出平台可提交的檔案（acct,label），並可選本地 F1 評估

0. 需求與安裝

Python 3.9+

套件：

pip install pandas numpy xgboost scikit-learn joblib


（選配）GPU 訓練：安裝可用 CUDA 的 xgboost

（選配）GPU 產特徵：RAPIDS cuDF（已在程式中自動偵測，安裝不到也能跑 CPU）

1. 資料放哪裡
dataset/
├─ acct_transaction.csv   # 交易明細（必要）
├─ acct_alert.csv         # 已知警示帳戶名單（訓練標記用）
└─ acct_predict.csv       # 要推論的帳號清單（可選；若有會優先用）


acct_transaction.csv 欄位只要能被自動偵測就行：

必要：from、to、amount（多種同義欄位會自動辨識）

可選：date/time/datetime、channel、currency、帳號型別(from/to)、是否自轉(is_self_txn)

2. 產生帳號特徵（account_features.py）

把交易逐筆資料轉成「每個帳號一列」的特徵表，含出金/入金聚合、夜間/週末、自轉帳、對手方多樣性、出入比例；
可選計算較昂貴的時間序列特徵（短時爆發度、到達間隔）。

最小指令：

python account_features.py --txns dataset/acct_transaction.csv --out feature_data/account_features.csv


常用參數：

--date-format / --time-format / --datetime-format：提供能減少解析警告

--gpu：若環境有 cuDF，開啟 GPU 加速 groupby

--time-stats：計算爆發度與到達間隔（大型資料較慢，預設關）

輸出：

feature_data/account_features.csv


第一欄為 acct。其餘為特徵（out_* 出金、in_* 入金、比率/旗標等）

3. 訓練與推論（main.py）

main.py 會：

把 特徵 與 警示名單 合併成 train.csv

呼叫 model_xgb.py 訓練 XGBoost

決定 test 檔來源 → 執行推論

輸出 submission.csv + 平台可交的 submission_final.csv (acct,label)

（選）與 acct_alert.csv 比對做本地 F1

一條龍（產訓練檔 → 訓練 → 推論 → 產提交）：

python main.py \
  --features feature_data/account_features.csv \
  --alerts dataset/acct_alert.csv \
  --outdir outputs


常用選項：

--skip-prepare：略過由 features+alerts 產 train.csv（若你已經有）

--test-csv：自行指定測試檔（預設流程：若找到 dataset/acct_predict.csv 會優先用；否則用全量 features 產 working/test_all.csv）

--model-path：模型存放路徑（預設 model_xgb.joblib）

--predict-only：只推論不訓練（會沿用 --model-path）

--eval：推論後用 dataset/acct_alert.csv 計算本地 F1（僅供参考）

推論控制（門檻/陽性率對齊平台很重要）：

--target-pos-rate：目標陽性率（例：0.002 = 0.2%），會用 分位數 自動選門檻

--min-positives：保底最少陽性數（避免全 0，最後用 Top-K 補）

--extra-thresholds：一次觀察多個門檻（逗號分隔，會列印每個門檻陽性比例）

--topk-show：列印機率最高前 K 筆，利於人工檢查

輸出：

outputs/submission.csv         # acct, proba, pred, pred_t0_5 ...（含診斷門檻欄）
outputs/submission_final.csv   # acct, label（平台可交）
working/train.csv              # features ⨝ alerts 的訓練表（若未 --skip-prepare）
