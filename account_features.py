# account_features.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# ====== 嘗試啟用 RAPIDS cuDF（GPU 加速）======
try:
    import cudf  # type: ignore
    _HAS_CUDF = True
except Exception:
    _HAS_CUDF = False


# -----------------------
# 欄位偵測
# -----------------------
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_schema(df: pd.DataFrame) -> Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]:
    """
    偵測欄位:
      from_acct / to_acct / 金額 / 日期 / 時間 / 來源(管道)
    會盡量對齊你給的 schema:
      from_acct, from_acct_type, to_acct, to_acct_type, is_self_txn,
      txn_amt, txn_date, txn_time, currency_type, channel_type
    """
    from_col = _find_col(df, ["from_acct", "from", "src_acct", "sender", "payer"])
    to_col   = _find_col(df, ["to_acct", "to", "dst_acct", "receiver", "payee"])
    amt_col  = _find_col(df, ["txn_amt", "amount", "amt", "money", "value"])
    date_col = _find_col(df, ["txn_date", "date"])
    time_col = _find_col(df, ["txn_time", "time", "ts", "timestamp", "datetime"])
    src_col  = _find_col(df, ["channel_type", "source", "channel", "src", "method"])

    if not (from_col and to_col and amt_col):
        raise ValueError("交易檔需至少包含 from_acct / to_acct / amount（含常見同義字）")

    return from_col, to_col, amt_col, date_col, time_col, src_col


def _build_timestamp(df: pd.DataFrame, date_col: Optional[str], time_col: Optional[str]) -> Optional[str]:
    """
    盡量組合 datetime 欄位：
      - 同時有日期與時間 => 合併為一個 'ts'
      - 只有一個 => 直接轉 'ts'
      - 都沒有 => 回傳 None
    """
    if date_col is None and time_col is None:
        return None

    if date_col is not None and time_col is not None:
        ts = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
    else:
        # 只靠單一欄位轉
        col = date_col if date_col is not None else time_col
        ts = pd.to_datetime(df[col], errors="coerce")

    df["ts"] = ts
    return "ts"


# -----------------------
# Pandas 聚合（CPU 後備）
# -----------------------
def _pd_group_amount_block(g: pd.core.groupby.generic.SeriesGroupBy, prefix: str) -> pd.DataFrame:
    basic = g.agg(["size", "sum", "mean", "std", "min", "max"]).rename(columns={
        "size": f"{prefix}_txn_count",
        "sum":  f"{prefix}_amt_sum",
        "mean": f"{prefix}_amt_mean",
        "std":  f"{prefix}_amt_std",
        "min":  f"{prefix}_amt_min",
        "max":  f"{prefix}_amt_max",
    })
    # 分位數（在 GPU 路徑為求穩定與速度省略；CPU 路徑保留）
    q = g.quantile([0.5, 0.9, 0.99]).unstack()
    q.columns = [f"{prefix}_amt_p{int(qv*100)}" for qv in q.columns]
    return basic.join(q, how="left")


# -----------------------
# cuDF 聚合（GPU 快速）
# -----------------------
def _cdf_group_agg(gdf: "cudf.DataFrame", key: str, amt_col: str,
                   is_night_col: str, is_wkend_col: str, is_self_col: str,
                   counterparty_col: str, src_col: Optional[str],
                   prefix: str) -> pd.DataFrame:
    """
    使用 cuDF 做高速聚合：count / sum / mean / std / min / max / 各種 nunique / sum of flags
    注意：為了速度與穩定性，這裡不計算分位數（p50/p90/p99）。
    """
    agg_dict = {
        amt_col: ["count", "sum", "mean", "std", "min", "max"],
        is_night_col: ["sum"],
        is_wkend_col: ["sum"],
        is_self_col: ["sum"],
        counterparty_col: ["nunique"],
    }
    if src_col:
        agg_dict[src_col] = ["nunique"]

    got = gdf.groupby(key).agg(agg_dict)

    # 攤平 MultiIndex 欄名並加上 prefix
    got = got.to_pandas()
    got.columns = [
        f"{prefix}_{c2 if c2!='count' else 'txn_count'}" if c1 == amt_col
        else (f"{prefix}_night_txn_count" if c1 == is_night_col
              else f"{prefix}_weekend_txn_count" if c1 == is_wkend_col
              else f"{prefix}_self_txn_count" if c1 == is_self_col
              else f"{prefix}_unique_to_acct" if c1 == counterparty_col and c2 == "nunique"
              else (f"{prefix}_source_nunique" if src_col and c1 == src_col and c2 == "nunique" else f"{prefix}_{c1}_{c2}"))
        for (c1, c2) in got.columns
    ]

    # 重新命名金額統計欄位：sum/mean/std/min/max
    ren = {
        f"{prefix}_{amt_col}_sum":  f"{prefix}_amt_sum",
        f"{prefix}_{amt_col}_mean": f"{prefix}_amt_mean",
        f"{prefix}_{amt_col}_std":  f"{prefix}_amt_std",
        f"{prefix}_{amt_col}_min":  f"{prefix}_amt_min",
        f"{prefix}_{amt_col}_max":  f"{prefix}_amt_max",
    }
    if f"{prefix}_{amt_col}_txn_count" in got.columns:
        ren[f"{prefix}_{amt_col}_txn_count"] = f"{prefix}_txn_count"
    got = got.rename(columns=ren)

    got.index.name = "acct"
    return got


# -----------------------
# 邏輯主體
# -----------------------
def build_account_features(
    txns_path: str = "dataset/acct_transaction.csv",
    out_path: str = "feature_data/account_features.csv",
    alerts_path: Optional[str] = "dataset/acct_alert.csv",
    use_gpu: bool = True,
) -> pd.DataFrame:
    """
    讀取原始交易，輸出帳號層級特徵。
    • GPU（cuDF）可用時，重度 groupby 全部走 GPU，速度更快。
    • 弱化昂貴特徵（例如 2-hop），保留「對 alert 的直接連結」與「鄰居中 alert 比例」（可高效完成）。
    • 針對極度不平衡資料：保留能凸顯異常的比率／稀疏統計（如夜間比率、來源多樣性、出入比例）。
    """
    df = pd.read_csv(txns_path)

    # 欄位對齊
    from_col, to_col, amt_col, date_col, time_col, src_col = _detect_schema(df)
    ts_col = _build_timestamp(df, date_col, time_col)

    # 基礎欄位清理
    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
    # is_self_txn：若原檔有就用原檔，否則用 from==to 推出
    if "is_self_txn" in df.columns:
        is_self_col = "is_self_txn"
        df[is_self_col] = df[is_self_col].fillna(0).astype(int)
    else:
        is_self_col = "_is_self_txn"
        df[is_self_col] = (df[from_col].astype(str) == df[to_col].astype(str)).astype(int)

    # 夜間 / 週末
    if ts_col:
        hour = df[ts_col].dt.hour
        dow  = df[ts_col].dt.dayofweek
        df["_is_night"]   = ((hour >= 22) | (hour < 6)).astype("int32")
        df["_is_weekend"] = (dow >= 5).astype("int32")
    else:
        df["_is_night"] = 0
        df["_is_weekend"] = 0

    # === 優先嘗試 GPU ===
    use_gpu = bool(use_gpu and _HAS_CUDF)
    if use_gpu:
        gdf = cudf.from_pandas(df[[from_col, to_col, amt_col, is_self_col, "_is_night", "_is_weekend"] + ([src_col] if src_col else [])])

        # 出金（from 分組）
        out_df = _cdf_group_agg(
            gdf=gdf,
            key=from_col,
            amt_col=amt_col,
            is_night_col="_is_night",
            is_wkend_col="_is_weekend",
            is_self_col=is_self_col,
            counterparty_col=to_col,
            src_col=src_col,
            prefix="out",
        )

        # 入金（to 分組）
        in_df = _cdf_group_agg(
            gdf=gdf.rename(columns={from_col: "_from", to_col: "_to"}),
            key="_to",
            amt_col=amt_col,
            is_night_col="_is_night",
            is_wkend_col="_is_weekend",
            is_self_col=is_self_col,
            counterparty_col="_from",
            src_col=src_col,
            prefix="in",
        )

    else:
        # ===== CPU 後備（含分位數）=====
        # 出金
        g_out_amt = df.groupby(from_col, sort=False)[amt_col]
        out_block = _pd_group_amount_block(g_out_amt, "out")
        out_extra = pd.DataFrame({
            "out_night_txn_count":   df.groupby(from_col, sort=False)["_is_night"].sum(),
            "out_weekend_txn_count": df.groupby(from_col, sort=False)["_is_weekend"].sum(),
            "out_self_txn_count":    df.groupby(from_col, sort=False)[is_self_col].sum(),
            "out_unique_to_acct":    df.groupby(from_col, sort=False)[to_col].nunique(),
        })
        if src_col:
            out_extra["out_source_nunique"] = df.groupby(from_col, sort=False)[src_col].nunique()
        out_df = out_block.join(out_extra, how="left")
        out_df.index.name = "acct"

        # 入金
        g_in_amt = df.groupby(to_col, sort=False)[amt_col]
        in_block = _pd_group_amount_block(g_in_amt, "in")
        in_extra = pd.DataFrame({
            "in_night_txn_count":   df.groupby(to_col, sort=False)["_is_night"].sum(),
            "in_weekend_txn_count": df.groupby(to_col, sort=False)["_is_weekend"].sum(),
            "in_self_txn_count":    df.groupby(to_col, sort=False)[is_self_col].sum(),
            "in_unique_from_acct":  df.groupby(to_col, sort=False)[from_col].nunique(),
        })
        if src_col:
            in_extra["in_source_nunique"] = df.groupby(to_col, sort=False)[src_col].nunique()
        in_df = in_block.join(in_extra, how="left")
        in_df.index.name = "acct"

    # 匯總為帳號層級
    out_df.index.name = "acct"
    in_df.index.name  = "acct"
    feat = out_df.add(in_df, fill_value=0)

    # 比率特徵（對不平衡較友善，能凸顯異常行為）
    feat["night_txn_ratio"] = (
        (feat.get("out_night_txn_count", 0) + feat.get("in_night_txn_count", 0)) /
        (feat.get("out_txn_count", 0) + feat.get("in_txn_count", 0) + 1e-9)
    )
    feat["weekend_txn_ratio"] = (
        (feat.get("out_weekend_txn_count", 0) + feat.get("in_weekend_txn_count", 0)) /
        (feat.get("out_txn_count", 0) + feat.get("in_txn_count", 0) + 1e-9)
    )
    feat["self_txn_count"] = feat.get("out_self_txn_count", 0) + feat.get("in_self_txn_count", 0)
    feat["out_in_txn_ratio"] = feat.get("out_txn_count", 0) / (feat.get("in_txn_count", 0) + 1e-9)
    feat["out_in_amt_ratio"] = feat.get("out_amt_sum", 0.0) / (feat.get("in_amt_sum", 0.0) + 1e-9)

    # ===== 與警示帳號的直接關聯（高效版本）=====
    if alerts_path and Path(alerts_path).exists():
        alerts = pd.read_csv(alerts_path)
        if "acct" not in alerts.columns:
            for c in ["acct_id", "account", "account_id", "id"]:
                if c in alerts.columns:
                    alerts = alerts.rename(columns={c: "acct"})
                    break
        alert_set = set(alerts["acct"].astype(str).unique())

        df["_from_s"] = df[from_col].astype(str)
        df["_to_s"]   = df[to_col].astype(str)

        # 直接連到 alert 的次數（出 / 入）
        direct_to_alert   = df[df["_to_s"].isin(alert_set)].groupby("_from_s").size().rename("out_to_alert_count")
        direct_from_alert = df[df["_from_s"].isin(alert_set)].groupby("_to_s").size().rename("in_from_alert_count")

        # 鄰居中 alert 比例（用 unique 對手方計算）
        out_neighbors = df.groupby("_from_s")["_to_s"].nunique().rename("out_unique_to_acct2")
        out_neighbors_alert = df[df["_to_s"].isin(alert_set)].groupby("_from_s")["_to_s"].nunique().rename("out_alert_to_unique")
        in_neighbors  = df.groupby("_to_s")["_from_s"].nunique().rename("in_unique_from_acct2")
        in_neighbors_alert = df[df["_from_s"].isin(alert_set)].groupby("_to_s")["_from_s"].nunique().rename("in_alert_from_unique")

        nei = pd.concat([out_neighbors, out_neighbors_alert, in_neighbors, in_neighbors_alert], axis=1).fillna(0)
        nei["alert_neighbor_ratio_out"] = nei["out_alert_to_unique"] / (nei["out_unique_to_acct2"] + 1e-9)
        nei["alert_neighbor_ratio_in"]  = nei["in_alert_from_unique"] / (nei["in_unique_from_acct2"] + 1e-9)
        nei = nei[["alert_neighbor_ratio_out", "alert_neighbor_ratio_in"]]
        nei.index.name = "acct"

        # 併入
        direct_to_alert.index.name = "acct"
        direct_from_alert.index.name = "acct"
        feat = feat.join(direct_to_alert, how="left").join(direct_from_alert, how="left").join(nei, how="left")
        for c in ["out_to_alert_count", "in_from_alert_count", "alert_neighbor_ratio_out", "alert_neighbor_ratio_in"]:
            if c in feat.columns:
                feat[c] = feat[c].fillna(0)

    # 清理 & 輸出
    feat = feat.reset_index()
    num_cols = feat.select_dtypes(include=[np.number]).columns
    feat[num_cols] = feat[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ 已輸出特徵檔：{out_path}  （{feat.shape[0]} rows, {feat.shape[1]} cols）")

    return feat


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build account-level features from acct_transaction.csv")
    p.add_argument("--txns", default="dataset/acct_transaction.csv")
    p.add_argument("--out",  default="feature_data/account_features.csv")
    p.add_argument("--alerts", default="dataset/acct_alert.csv")
    p.add_argument("--no-gpu", action="store_true", help="強制使用 CPU（關閉 cuDF）")
    return p.parse_args()


def main():
    args = parse_args()
    build_account_features(
        txns_path=args.txns,
        out_path=args.out,
        alerts_path=args.alerts,
        use_gpu=not args.no_gpu,   # 預設開 GPU（若安裝 cuDF）
    )


if __name__ == "__main__":
    main()
