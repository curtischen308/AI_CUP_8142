# model_xgb.py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from joblib import dump, load
from pathlib import Path


def _encode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """將 object/string 欄位以 factorize 轉成整數，避免 XGBoost DMatrix 報 dtype 錯。"""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.factorize(df[col])[0]
    return df


def _pick_device_params():
    """
    XGBoost 2.x 建議：GPU 用 device='cuda' + tree_method='hist'
    若不可用則退回 CPU。
    """
    try:
        params = {"objective": "binary:logistic", "tree_method": "hist", "device": "cuda"}
        dm = xgb.DMatrix(np.array([[0.0], [1.0]]), label=np.array([0, 1]))
        xgb.train(params, dm, num_boost_round=1)
        return {"tree_method": "hist", "device": "cuda"}
    except Exception:
        return {"tree_method": "hist", "device": "cpu"}


def train_model(train_path="train.csv", model_path="model_xgb.joblib"):
    data = pd.read_csv(train_path)

    if "acct" not in data.columns or "label" not in data.columns:
        raise ValueError("train.csv 必須包含 'acct' 與 'label' 欄位")

    X = data.drop(columns=["acct", "label"]).copy()
    y = data["label"].astype(int)

    # object → 整數；NaN/Inf→0
    X = _encode_object_columns(X)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0:
        raise ValueError("訓練資料沒有正樣本，無法訓練。")
    scale = max(1.0, neg / max(1, pos))
    print(f"正樣本: {pos}, 負樣本: {neg}, scale_pos_weight={scale:.2f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X.columns.tolist())
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=X.columns.tolist())

    dev = _pick_device_params()
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],   # 對不平衡更敏感
        "eta": 0.09,                           # 降學習率，提升泛化
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "scale_pos_weight": scale,
        "max_bin": 256,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        **dev,
    }
    print(f"使用 device={dev['device']} tree_method={dev['tree_method']}")

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=50,
    )

    # PR 曲線挑最佳 F1（與 threshold 對齊）
    proba = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
    prec, rec, thr = precision_recall_curve(y_val, proba)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    best_thr = float(thr[best_idx])
    print(f"最佳閾值={best_thr:.4f}, 驗證集 F1={f1s[best_idx]:.4f}")

    # 存模型 + 特徵名 + 閾值
    dump({"model": bst, "threshold": best_thr, "feature_cols": X.columns.tolist()}, model_path)
    print(f"✅ 模型已存檔到 {model_path}")

    # 存特徵重要度（方便你調參/選特徵）
    try:
        fmap = pd.DataFrame({
            "feature": X.columns,
            "gain":    bst.get_score(importance_type="gain"),
            "weight":  bst.get_score(importance_type="weight"),
            "cover":   bst.get_score(importance_type="cover"),
        }).fillna(0)
        fmap.to_csv("outputs/feature_importance.csv", index=False, encoding="utf-8")
        print("📎 已輸出特徵重要度到 outputs/feature_importance.csv")
    except Exception:
        pass

def predict(
    test_path: str = "test.csv",
    model_path: str = "model_xgb.joblib",
    output_path: str = "prediction.csv",
    thresholds: list[float] | None = None,
    topk_show: int = 10
):
    """
    推論 + 診斷輸出
    - thresholds: 額外想看的多組閾值（會印出每個閾值的陽性數與比例，並在 CSV 加欄位 pred_tXX）
    - topk_show: 會印出機率最高的前 k 筆，方便肉眼檢查
    """
    if thresholds is None:
        thresholds = [0.5, 0.3, 0.2, 0.1]

    bundle = load(model_path)
    bst = bundle["model"]
    best_thr = float(bundle["threshold"])
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(test_path)

    # 容錯：acct / acct_id / account / account_id / id 皆可
    if "acct" not in df.columns:
        for c in ["acct_id", "account", "account_id", "id"]:
            if c in df.columns:
                df = df.rename(columns={c: "acct"})
                break
    if "acct" not in df.columns:
        raise ValueError("測試檔必須包含 acct（或 acct_id/account/account_id/id 其中之一）")

    # 對齊特徵（缺的補 0，多的丟掉）
    X_new = df.reindex(columns=feature_cols, fill_value=0)
    X_new = _encode_object_columns(X_new)
    X_new = X_new.replace([np.inf, -np.inf], np.nan).fillna(0)

    dnew = xgb.DMatrix(X_new)
    proba = bst.predict(dnew)

    # === 診斷：分數統計 ===
    print("\n📈 Score stats (prediction probabilities)")
    print(f"min={proba.min():.6f}  p50={np.median(proba):.6f}  mean={proba.mean():.6f}  "
          f"p90={np.quantile(proba, 0.9):.6f}  max={proba.max():.6f}")

    # 使用訓練時的最佳閾值（主輸出）
    pred = (proba >= best_thr).astype(int)
    print(f"\n🎯 Using best threshold from training: thr={best_thr:.4f}")
    print(f"positives={pred.sum()} / {len(pred)}  ({pred.mean()*100:.3f}%)")

    # 額外多組閾值
    extra_cols = {}
    print("\n🔎 Extra thresholds inspection:")
    for thr in thresholds:
        p = (proba >= thr).astype(int)
        ratio = p.mean() * 100
        print(f" - thr={thr:>5.2f} -> positives={p.sum():>6} / {len(p)}  ({ratio:6.3f}%)")
        extra_cols[f"pred_t{str(thr).replace('.','_')}"] = p

    # 機率最高的前 k 筆
    if topk_show > 0:
        top_idx = np.argsort(-proba)[:topk_show]
        print(f"\n👀 Top {topk_show} by probability:")
        for i in top_idx:
            print(f"  acct={df.loc[i, 'acct']}, proba={proba[i]:.6f}")

    # === 輸出 CSV ===
    out = pd.DataFrame({"acct": df["acct"], "proba": proba, "pred": pred})
    for c, v in extra_cols.items():
        out[c] = v

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✅ 預測輸出到 {output_path}")