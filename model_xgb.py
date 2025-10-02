# run_experiments.py
from __future__ import annotations
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve

# ---------------- Utils ----------------
def _encode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把 object 類型用 factorize 轉成整數，避免 DMatrix dtype 問題。"""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.factorize(df[c])[0]
    return df

def _pick_device_params() -> Dict[str, str]:
    """優先用 GPU，無法使用就退回 CPU。"""
    try:
        params = {"objective": "binary:logistic", "tree_method": "hist", "device": "cuda"}
        dm = xgb.DMatrix(np.array([[0.0], [1.0]]), label=np.array([0, 1]))
        xgb.train(params, dm, num_boost_round=1)
        return {"tree_method": "hist", "device": "cuda"}
    except Exception:
        return {"tree_method": "hist", "device": "cpu"}

def _best_f1_from_pr(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float, float, float]:
    """
    從 PR 曲線找 F1* 與對應 threshold、P、R。
    回傳: (best_f1, best_thr, best_precision, best_recall)
    """
    precision, recall, thr = precision_recall_curve(y_true, proba)
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    idx = int(np.nanargmax(f1s))
    return float(f1s[idx]), float(thr[idx]), float(precision[idx]), float(recall[idx])

def _prepare_xy(features_path: str, alerts_path: str, mode: str) -> Tuple[pd.DataFrame, pd.Series]:
    """由 features + alerts 準備 X, y；mode ∈ {'in','out','both'} 控制取哪些特徵。"""
    feat = pd.read_csv(features_path)
    alerts = pd.read_csv(alerts_path)

    # 對齊 acct 欄位
    def _acct_c(df: pd.DataFrame, candidates=None):
        candidates = candidates or ["acct", "acct_id", "account", "account_id", "id"]
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError("無 acct 欄位")

    feat = feat.rename(columns={_acct_c(feat, ["acct","acct_id","account","id"]): "acct"})
    alerts = alerts.rename(columns={_acct_c(alerts, ["acct","acct_id","account","account_id","id"]): "acct"})
    alerts = alerts.drop_duplicates(subset=["acct"]).copy()
    alerts["label"] = 1

    df = feat.merge(alerts[["acct","label"]], on="acct", how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    # 依 mode 篩特徵
    ALL = [c for c in df.columns if c not in ("acct","label")]
    IN  = [c for c in ALL if c.startswith("in_") or c.startswith("in") and ("_w" in c or "_iat" in c or "burst" in c)]
    OUT = [c for c in ALL if c.startswith("out_") or c.startswith("out") and ("_w" in c or "_iat" in c or "burst" in c)]

    # 更穩健一點：用包含字眼來挑
    def _pick(cols: List[str], keywords: List[str]) -> List[str]:
        sel = []
        for c in cols:
            cc = c.lower()
            if any(k in cc for k in keywords):
                sel.append(c)
        return sorted(set(sel))

    in_keys  = ["in_", "in", "night", "weekend", "iat", "burst", "unique_from", "amt_", "txn_count"]
    out_keys = ["out_", "out", "night", "weekend", "iat", "burst", "unique_to", "amt_", "txn_count"]

    in_feats  = _pick(ALL, in_keys)
    out_feats = _pick(ALL, out_keys)

    if mode == "in":
        cols = in_feats
    elif mode == "out":
        cols = out_feats
    elif mode == "both":
        cols = sorted(set(in_feats) | set(out_feats))
    else:
        raise ValueError("mode must be in/out/both")

    X = df[cols].copy()
    y = df["label"].copy()
    return X, y

@dataclass
class ExpResult:
    mode: str
    imb: str
    n_feat: int
    ap: float
    f1_star: float
    thr: float
    precision: float
    recall: float
    pos_at_best_pct: float
    best_iteration: int
    device: str

# ---------------- Core ----------------
def _train_one(X: pd.DataFrame, y: pd.Series, imbalance: str) -> Tuple[xgb.Booster, Dict]:
    """
    訓練一個模型；imbalance ∈ {'sqrt','full'} 控制 scale_pos_weight。
    回傳 booster 與訓練資訊（含 best_iteration）
    """
    X = _encode_object_columns(X).replace([np.inf, -np.inf], np.nan).fillna(0)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale_full = max(1.0, neg / max(1, pos))
    scale_sqrt = float(np.sqrt(scale_full))
    scale = scale_sqrt if imbalance == "sqrt" else scale_full

    Xtr, Xva, ytr, yva = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=X.columns.tolist())
    dva = xgb.DMatrix(Xva, label=yva, feature_names=X.columns.tolist())

    dev = _pick_device_params()
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": 0.08,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale,
        "max_bin": 256,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        **dev,
    }
    print(f"🧪 device={dev['device']} tree_method={dev['tree_method']}  |  scale_pos_weight={scale:.2f}")

    bst = xgb.train(
        params,
        dtr,
        num_boost_round=1000,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=80,
        verbose_eval=100,
    )
    info = {
        "best_iteration": int(bst.best_iteration),
        "device": dev["device"],
        "X_val": Xva,
        "y_val": yva,
        "dval": dva,
    }
    return bst, info

def _eval_one(bst: xgb.Booster, info: Dict) -> Tuple[float, float, float, float, float, float]:
    """計算 AP 與最佳 F1（含 thr / P / R / pos%）。"""
    dva = info["dval"]
    y_val = info["y_val"].values
    proba = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))

    ap = average_precision_score(y_val, proba)
    f1_star, thr, prec, rec = _best_f1_from_pr(y_val, proba)
    pos_rate = float((proba >= thr).mean() * 100.0)
    return ap, f1_star, thr, prec, rec, pos_rate

def run_experiments(features_path: str, alerts_path: str, out_dir: str, sort_by: str = "ap"):
    """
    跑 6 組：mode ∈ {in, out, both} × imb ∈ {sqrt, full}
    產出 outputs/exp_summary.csv 並印表格；sort_by 可選 'ap' 或 'f1_star'
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results: List[ExpResult] = []

    for mode in ["in", "out", "both"]:
        # 準備資料
        X, y = _prepare_xy(features_path, alerts_path, mode)

        print("Label 分布：\n", y.value_counts())
        print(f"DEBUG | mode={mode} | X type={type(X)} shape={X.shape} dtypes={X.dtypes.nunique()}")

        for imb in ["sqrt", "full"]:
            bst, info = _train_one(X, y, imbalance=imb)
            ap, f1s, thr, p, r, pos_pct = _eval_one(bst, info)

            print(f"✓ 完成 mode={mode:>4}  imb={imb:<5}  AP={ap:.5f}  F1*={f1s:.4f}  "
                  f"P={p:.4f} R={r:.4f}  pos@best={pos_pct:.4f}%")

            res = ExpResult(
                mode=mode,
                imb=imb,
                n_feat=X.shape[1],
                ap=ap,
                f1_star=f1s,
                thr=thr,
                precision=p,
                recall=r,
                pos_at_best_pct=pos_pct,
                best_iteration=info["best_iteration"],
                device=info["device"],
            )
            results.append(res)

    # 彙整表
    df_res = pd.DataFrame([asdict(r) for r in results])
    sort_key = "ap" if sort_by.lower() == "ap" else "f1_star"
    df_res_sorted = df_res.sort_values(sort_key, ascending=False).reset_index(drop=True)

    # 儲存 CSV
    out_csv = Path(out_dir) / "exp_summary.csv"
    df_res_sorted.to_csv(out_csv, index=False, encoding="utf-8")

    # 顯示表格
    print("\n================== 實驗彙整（排序依 {}） ==================".format(sort_key.upper()))
    print(df_res_sorted.to_string(index=False, formatters={
        "ap": lambda v: f"{v:.5f}",
        "f1_star": lambda v: f"{v:.4f}",
        "thr": lambda v: f"{v:.6f}",
        "precision": lambda v: f"{v:.4f}",
        "recall": lambda v: f"{v:.4f}",
        "pos_at_best_pct": lambda v: f"{v:.4f}%",
    }))
    best = df_res_sorted.iloc[0].to_dict()
    print("\n🏆 最佳組合 -> mode={mode}, imb={imb}, AP={ap:.5f}, F1*={f1_star:.4f}, "
          "P={precision:.4f}, R={recall:.4f}, thr={thr:.6f}, pos@best={pos_at_best_pct:.4f}%"
          .format(**best))
    print(f"📄 已儲存彙整表：{out_csv}")

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Run ablation experiments and summarize results.")
    p.add_argument("--features", default="feature_data/account_features.csv")
    p.add_argument("--alerts",   default="dataset/acct_alert.csv")
    p.add_argument("--out",      default="outputs")
    p.add_argument("--sort-by",  default="ap", choices=["ap","f1"], help="彙整表排序依據")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sort_by = "ap" if args.sort_by.lower() == "ap" else "f1"
    run_experiments(args.features, args.alerts, args.out, sort_by=("ap" if sort_by=="ap" else "f1_star"))
