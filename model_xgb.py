# model_xgb.py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score
from joblib import dump, load
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def _encode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把 object 類型用 factorize 轉成整數，避免 DMatrix dtype 問題。"""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.factorize(df[col])[0]
    return df


def _pick_device_params():
    """
    優先用 GPU（device='cuda' + tree_method='hist'），不可用才退回 CPU。
    """
    try:
        params = {"objective": "binary:logistic", "tree_method": "hist", "device": "cuda"}
        dm = xgb.DMatrix(np.array([[0.0], [1.0]]), label=np.array([0, 1]))
        xgb.train(params, dm, num_boost_round=1)
        return {"tree_method": "hist", "device": "cuda"}
    except Exception:
        return {"tree_method": "hist", "device": "cpu"}


def _choose_threshold(y_true: np.ndarray, proba: np.ndarray, pos_rate_train: float) -> float:
    """
    智慧選 threshold，避免全 0：
      1) 先用 PR 曲線選 F1 最佳
      2) 若仍全 0，退回以訓練集陽性率為目標的分位數閾值（*1.5 buffer）
    """
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)

    best_idx = int(np.nanargmax(f1s))
    best_thr = float(thr[best_idx])

    pred_a = (proba >= best_thr).astype(int)
    if pred_a.sum() == 0:
        q = max(1e-6, min(0.02, pos_rate_train * 1.5))
        best_thr = float(np.quantile(proba, 1 - q))

    return best_thr


def _random_undersample(X: pd.DataFrame, y: pd.Series, max_neg_per_pos: int = 500, seed: int = 42):
    """
    在極端不平衡情況下，做簡單隨機下採樣（不需額外套件）以加速&穩定。
    """
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y

    cap = int(min(len(neg_idx), max_neg_per_pos * max(1, len(pos_idx))))
    if cap >= len(neg_idx):
        return X, y

    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg_idx, size=cap, replace=False)
    keep_idx = np.concatenate([pos_idx.values, keep_neg])
    keep_idx.sort()
    return X.loc[keep_idx], y.loc[keep_idx]


def train_model(train_path="train.csv", model_path="model_xgb.joblib"):
    data = pd.read_csv(train_path)

    if "acct" not in data.columns or "label" not in data.columns:
        raise ValueError("train.csv 必須包含 'acct' 與 'label' 欄位")

    X_full = data.drop(columns=["acct", "label"]).copy()
    y_full = data["label"].astype(int)

    # 整理特徵
    X_full = _encode_object_columns(X_full)
    X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(0)

    pos, neg = int((y_full == 1).sum()), int((y_full == 0).sum())
    if pos == 0:
        raise ValueError("訓練資料沒有正樣本，無法訓練。")
    pos_rate_train = pos / (pos + neg + 1e-9)
    scale = float(np.sqrt(neg / max(1, pos)))  # 比傳統 neg/pos 更溫和
    print(f"正樣本: {pos}, 負樣本: {neg}, pos_rate={pos_rate_train:.6f}, scale_pos_weight={scale:.2f}")

    # 下採樣（極端不平衡才啟動）
    if neg / max(1, pos) > 1000:
        X_bal, y_bal = _random_undersample(X_full, y_full, max_neg_per_pos=800)
        print(f"🔧 啟用隨機下採樣：{len(y_full)} -> {len(y_bal)}")
    else:
        X_bal, y_bal = X_full, y_full

    X_train, X_val, y_train, y_val = train_test_split(
        X_bal, y_bal, stratify=y_bal, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_bal.columns.tolist())
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=X_bal.columns.tolist())

    dev = _pick_device_params()
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",     # PR-AUC 對不平衡更敏感
        "eta": 0.1,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "scale_pos_weight": scale,
        "max_bin": 256,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        **dev,
    }
    print(f"使用 device={dev['device']} tree_method={dev['tree_method']}")

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=150,
        verbose_eval=100,
    )

    # 驗證集機率與閾值
    proba_val = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
    best_thr = _choose_threshold(y_val.values, proba_val, pos_rate_train)

    # 驗證指標
    y_pred_val = (proba_val >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_val).ravel()
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    pr_auc = roc_auc_score(y_val, proba_val)  # 這裡其實是 ROC-AUC；PR-AUC 請用 precision_recall_curve+auc
    print(f"選定閾值={best_thr:.6f} | 驗證: TP={tp} FP={fp} TN={tn} FN={fn} | P={prec:.4f} R={rec:.4f} F1={f1:.4f} | ROC-AUC={pr_auc:.4f}")

    # 存模型 + 閾值 + 特徵名 + 訓練陽性率
    dump({
        "model": bst,
        "threshold": best_thr,
        "feature_cols": X_bal.columns.tolist(),
        "pos_rate_train": pos_rate_train
    }, model_path)
    print(f"✅ 模型已存檔到 {model_path}")

    # 特徵重要度（方便調參/選特徵）
    try:
        fmap_gain   = bst.get_score(importance_type="gain")
        fmap_weight = bst.get_score(importance_type="weight")
        fmap_cover  = bst.get_score(importance_type="cover")
        all_feats = list({*X_bal.columns.tolist(), *fmap_gain.keys(), *fmap_weight.keys(), *fmap_cover.keys()})
        rows = []
        for f in all_feats:
            rows.append({
                "feature": f,
                "gain":   fmap_gain.get(f, 0.0),
                "weight": fmap_weight.get(f, 0.0),
                "cover":  fmap_cover.get(f, 0.0),
            })
        fmap = pd.DataFrame(rows)
        Path("outputs").mkdir(exist_ok=True, parents=True)
        fmap.sort_values("gain", ascending=False).to_csv("outputs/feature_importance.csv", index=False, encoding="utf-8")
        print("📎 已輸出特徵重要度到 outputs/feature_importance.csv")
    except Exception:
        pass


def predict(
    test_path: str = "test.csv",
    model_path: str = "model_xgb.joblib",
    output_path: str = "prediction.csv",
    thresholds: list[float] | None = None,
    topk_show: int = 10,
    target_pos_rate: float | None = None,
    min_positives: int = 1,  # 最少陽性數（最後保險）
):
    """
    推論 + 診斷輸出
    - thresholds: 額外想看的多組閾值（會印出每個閾值的陽性數與比例，並在 CSV 加欄位 pred_tXX）
    - topk_show: 會印出機率最高的前 k 筆，方便肉眼檢查
    - target_pos_rate: 想要的陽性比例，比如 0.002 -> 0.2%，會用分位數求 threshold 覆寫
    - min_positives: 最少陽性數（Top-K 保底）
    """
    if thresholds is None:
        thresholds = [0.5, 0.3, 0.2, 0.1]

    bundle = load(model_path)
    bst = bundle["model"]
    best_thr = float(bundle.get("threshold", 0.5))
    feature_cols = bundle["feature_cols"]
    pos_rate_train = float(bundle.get("pos_rate_train", 0.0005))

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
    for col in X_new.columns:
        if X_new[col].dtype == "object":
            X_new[col] = pd.factorize(X_new[col])[0]
    X_new = X_new.replace([np.inf, -np.inf], np.nan).fillna(0)

    dnew = xgb.DMatrix(X_new, feature_names=feature_cols)
    try:
        proba = bst.predict(dnew, iteration_range=(0, bst.best_iteration + 1))
    except Exception:
        proba = bst.predict(dnew)

    # === 診斷：分數分佈 ===
    print("\n📈 Score stats (prediction probabilities)")
    print(f"min={proba.min():.6f}  p50={np.median(proba):.6f}  mean={proba.mean():.6f}  "
          f"p90={np.quantile(proba, 0.9):.6f}  max={proba.max():.6f}")

    # 1) 預設用訓練得到的最佳 threshold
    use_thr = float(best_thr)

    # 2) 若有指定目標陽性率，改用分位數覆寫 threshold
    if target_pos_rate is not None and target_pos_rate > 0:
        q = max(1e-5, min(0.05, float(target_pos_rate)))
        use_thr = float(np.quantile(proba, 1 - q))
        print(f"🔧 覆寫 threshold 以達目標陽性率 ~{q*100:.3f}% -> thr={use_thr:.6f}")

    pred = (proba >= use_thr).astype(int)
    print(f"\n🎯 Using threshold: thr={use_thr:.6f}")
    print(f"positives={pred.sum()} / {len(pred)}  ({pred.mean()*100:.3f}%)")

    # 3) 若仍過度保守（幾乎全 0），按「訓練陽性率的 1.5 倍」分位數回退
    if pred.sum() < max(1, min_positives):
        fallback_rate = max(1e-6, min(0.05, pos_rate_train * 1.5))
        new_thr = float(np.quantile(proba, 1 - fallback_rate))
        pred_fb = (proba >= new_thr).astype(int)
        print(f"⚠️  全 0 風險，回退分位數 threshold：{new_thr:.6f} (target~{fallback_rate*100:.4f}%) "
              f"-> positives={pred_fb.sum()} ({pred_fb.mean()*100:.3f}%)")
        if pred_fb.sum() >= max(1, min_positives):
            use_thr = new_thr
            pred = pred_fb

    # 4) 若還是不夠，再強制取 Top-K（保底不為 0）
    if pred.sum() < max(1, min_positives) and proba.size > 0:
        need = max(1, min_positives) - int(pred.sum())
        top_idx = np.argsort(-proba)[:need]
        pred[top_idx] = 1
        print(f"🛟 強制取 Top-{need} 做陽性保底；new positives={pred.sum()} ({pred.mean()*100:.3f}%)")
        use_thr = float(min(proba[top_idx].min(), use_thr))

    # 額外多組閾值診斷
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
