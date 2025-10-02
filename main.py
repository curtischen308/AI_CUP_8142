# main.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from model_xgb import train_model, predict
from evaluate import evaluate_local_f1


# ---------------- Helpers ----------------
def _find_acct_col(df: pd.DataFrame, candidates=None) -> str:
    if candidates is None:
        candidates = ["acct_id", "account_id", "acct", "account", "id"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"無法找到帳號欄位，需包含其中之一: {candidates}")


def prepare_train_csv(features_path: Path, alerts_path: Path, out_csv: Path) -> None:
    features = pd.read_csv(features_path)
    alerts = pd.read_csv(alerts_path)

    feat_acct_col = _find_acct_col(features, ["acct_id", "acct", "account", "id"])
    alert_acct_col = _find_acct_col(alerts, ["acct_id", "account_id", "acct", "account", "id"])

    features = features.rename(columns={feat_acct_col: "acct"})
    alerts = alerts.rename(columns={alert_acct_col: "acct"}).drop_duplicates(subset=["acct"]).copy()
    alerts["label"] = 1

    df = features.merge(alerts[["acct", "label"]], on="acct", how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    cols = ["acct", "label"] + [c for c in df.columns if c not in ("acct", "label")]
    df = df[cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ 已輸出訓練檔: {out_csv}")


def decide_test_csv(args) -> str:
    if args.test_csv is not None:
        print(f"ℹ️  使用指定的測試檔：{args.test_csv}")
        return args.test_csv

    predict_candidate = Path("dataset/acct_predict.csv")
    if predict_candidate.exists():
        df_pred = pd.read_csv(predict_candidate)
        acct_col = _find_acct_col(df_pred, ["acct", "acct_id", "account", "account_id", "id"])
        if acct_col != "acct":
            df_pred = df_pred.rename(columns={acct_col: "acct"})
        cols = ["acct"] + [c for c in df_pred.columns if c != "acct"]
        df_pred = df_pred[cols]
        out_path = Path("working/test_from_acct_predict.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_pred.to_csv(out_path, index=False)
        print(f"ℹ️  偵測到 dataset/acct_predict.csv，將以此做推論：{out_path}")
        return str(out_path)

    feat = pd.read_csv(args.features)
    acct_col = _find_acct_col(feat, ["acct_id", "acct", "account", "id"])
    test_df = feat.rename(columns={acct_col: "acct"}).copy()
    out_path = Path("working/test_all.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_cols = ["acct"] + [c for c in test_df.columns if c != "acct"]
    test_df[test_cols].to_csv(out_path, index=False)
    print(f"ℹ️  未提供 test.csv，亦未找到 dataset/acct_predict.csv，已自動建立：{out_path}")
    return str(out_path)


def write_submission_final(submission_path: str, final_path: str):
    """輸出平台可交的兩欄檔：acct,label（把 pred 改名為 label）"""
    sub = pd.read_csv(submission_path)
    need_cols = {"acct", "pred"}
    if not need_cols.issubset(sub.columns):
        raise ValueError(f"submission.csv 必須包含欄位: {need_cols}")
    final = sub[["acct", "pred"]].rename(columns={"pred": "label"})
    final["label"] = final["label"].astype(int)
    Path(final_path).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(final_path, index=False, encoding="utf-8")
    print(f"✅ 產生平台提交檔：{final_path}")


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="AI CUP Orchestrator (XGB)")
    parser.add_argument("--features", default="feature_data/account_features.csv")
    parser.add_argument("--alerts",   default="dataset/acct_alert.csv")
    parser.add_argument("--outdir",   default="outputs")

    parser.add_argument("--train-csv",  default="working/train.csv")
    parser.add_argument("--test-csv",   default=None)
    parser.add_argument("--skip-prepare", action="store_true", help="略過由 features+alerts 產生 train.csv")

    parser.add_argument("--model-path", default="model_xgb.joblib")
    parser.add_argument("--predict-only", action="store_true", help="只做推論")
    parser.add_argument("--eval", action="store_true", help="推論後用 acct_alert.csv 計算本地 F1")

    # 推論控制（與 model_xgb.predict 對齊）
    parser.add_argument("--target-pos-rate", type=float, default=None,
                        help="指定目標陽性率（例如 0.002 代表 0.2%），將以分位數覆寫 threshold")
    parser.add_argument("--min-positives", type=int, default=1,
                        help="保底最少陽性數（Top-K 強制置 1）")
    parser.add_argument("--extra-thresholds", type=str, default="0.5,0.3,0.2,0.1",
                        help="額外檢視的 threshold 清單，逗號分隔")
    parser.add_argument("--topk-show", type=int, default=10,
                        help="診斷印出機率最高的前 K 筆")

    args = parser.parse_args()
    Path("working").mkdir(exist_ok=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_csv = Path(args.train_csv)
    if not args.skip_prepare and not args.predict_only:
        prepare_train_csv(Path(args.features), Path(args.alerts), train_csv)

    if not args.predict_only:
        train_model(str(train_csv), args.model_path)

    test_csv = decide_test_csv(args)
    submission_path = str(outdir / "submission.csv")

    # 解析 extra thresholds
    thr_list = []
    if args.extra_thresholds:
        for t in args.extra_thresholds.split(","):
            t = t.strip()
            if t:
                try:
                    thr_list.append(float(t))
                except Exception:
                    pass

    # 推論
    predict(
        test_path=test_csv,
        model_path=args.model_path,
        output_path=submission_path,
        thresholds=thr_list if thr_list else None,
        topk_show=args.topk_show,
        target_pos_rate=args.target_pos_rate,
        min_positives=args.min_positives,
    )
    print(f"✅ 完成推論，提交檔已輸出：{submission_path}")

    # 產生兩欄的提交檔（平台可交）
    write_submission_final(submission_path, str(outdir / "submission_final.csv"))

    if args.eval:
        evaluate_local_f1(submission_path, args.alerts)


if __name__ == "__main__":
    main()
