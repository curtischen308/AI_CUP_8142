# account_features.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# 可選 GPU：RAPIDS cuDF（若環境安裝 & 有 GPU）
try:
    import cudf  # type: ignore
    _HAS_CUDF = True
except Exception:
    _HAS_CUDF = False


# ============ 小工具 ============
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_schema(df: pd.DataFrame) -> Tuple[str, str, str,
                                              Optional[str], Optional[str], Optional[str],
                                              Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    偵測欄位：
      必要：from / to / amount
      可選：date / time / datetime / channel / currency / from_acct_type / to_acct_type / is_self_txn
    """
    from_col = _find_col(df, ["from_acct", "from", "src_acct", "sender", "payer"])
    to_col   = _find_col(df, ["to_acct", "to", "dst_acct", "receiver", "payee"])
    amt_col  = _find_col(df, ["txn_amt", "amount", "amt", "money", "value"])

    date_col = _find_col(df, ["txn_date", "date"])
    time_col = _find_col(df, ["txn_time", "time"])
    dt_col   = _find_col(df, ["txn_datetime", "datetime", "timestamp", "ts"])

    channel_col  = _find_col(df, ["channel_type", "channel", "source", "method"])
    currency_col = _find_col(df, ["currency_type", "currency", "curr"])
    from_type    = _find_col(df, ["from_acct_type", "from_type", "src_type"])
    to_type      = _find_col(df, ["to_acct_type", "to_type", "dst_type"])
    self_col     = _find_col(df, ["is_self_txn", "is_self", "self_txn"])

    if not (from_col and to_col and amt_col):
        raise ValueError("交易檔需包含 from_acct / to_acct / txn_amt（或同義欄位）。")

    return from_col, to_col, amt_col, date_col, time_col, dt_col, channel_col, currency_col, from_type, to_type, self_col


def _coerce_bool01(s: pd.Series) -> pd.Series:
    """
    把各式表示轉成 0/1：
    'Y/Yes/True/1' -> 1；'N/No/False/0' -> 0；其餘/NaN -> 0
    """
    if s.dtype.kind in ("i", "u", "b", "f"):
        return (pd.to_numeric(s, errors="coerce").fillna(0) != 0).astype("int32")
    m = {"y": 1, "yes": 1, "true": 1, "t": 1, "1": 1,
         "n": 0, "no": 0,  "false": 0, "f": 0, "0": 0}
    return s.astype(str).str.strip().str.lower().map(m).fillna(0).astype("int32")


def _combine_datetime(df: pd.DataFrame,
                      date_col: Optional[str], time_col: Optional[str], dt_col: Optional[str],
                      fmt_date: Optional[str], fmt_time: Optional[str], fmt_datetime: Optional[str]) -> Optional[pd.Series]:
    """
    解析時間（盡量避免 pandas 警告）：
      1) 若有 dt_col 且給了 fmt_datetime，先用它；失敗就退回寬鬆解析
      2) 否則若有 date+time，先用 fmt_date+fmt_time；失敗就退回寬鬆解析
      3) 否則回 None
    """
    if dt_col and dt_col in df.columns:
        if fmt_datetime:
            try:
                return pd.to_datetime(df[dt_col], format=fmt_datetime, errors="raise")
            except Exception:
                pass
        return pd.to_datetime(df[dt_col], errors="coerce")

    if date_col and time_col and (date_col in df.columns) and (time_col in df.columns):
        if fmt_date and fmt_time:
            try:
                return pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str),
                                      format=f"{fmt_date} {fmt_time}", errors="raise")
            except Exception:
                pass
        return pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")

    return None


def _group_amt_block_pd(df: pd.DataFrame, key: str, amt_col: str, prefix: str) -> pd.DataFrame:
    g = df.groupby(key, sort=False)[amt_col]
    basic = g.agg(['size', 'sum', 'mean', 'std', 'min', 'max']).rename(columns={
        'size': f"{prefix}_txn_count",
        'sum':  f"{prefix}_amt_sum",
        'mean': f"{prefix}_amt_mean",
        'std':  f"{prefix}_amt_std",
        'min':  f"{prefix}_amt_min",
        'max':  f"{prefix}_amt_max",
    })
    q = g.quantile([0.5, 0.9, 0.99]).unstack()
    q.columns = [f"{prefix}_amt_p{int(qv*100)}" for qv in q.columns]
    return basic.join(q, how="left")


def _group_amt_block_gpu(df: pd.DataFrame, key: str, amt_col: str, prefix: str) -> pd.DataFrame:
    """
    GPU 加速：size/sum/mean/std/min/max 用 cuDF；quantile 保留在 pandas（準確）
    """
    gdf = cudf.from_pandas(df[[key, amt_col]])
    gb = gdf.groupby(key)
    basic = gb.agg({amt_col: ["count", "sum", "mean", "std", "min", "max"]}).reset_index().to_pandas()
    basic.columns = [key, f"{prefix}_txn_count", f"{prefix}_amt_sum", f"{prefix}_amt_mean",
                     f"{prefix}_amt_std", f"{prefix}_amt_min", f"{prefix}_amt_max"]
    basic = basic.set_index(key)

    q = df.groupby(key, sort=False)[amt_col].quantile([0.5, 0.9, 0.99]).unstack()
    q.columns = [f"{prefix}_amt_p{int(qv*100)}" for qv in q.columns]
    return basic.join(q, how="left")


# ============ 主流程 ============
def build_account_features(
    txns_path: str = "dataset/acct_transaction.csv",
    out_path: str = "feature_data/account_features.csv",
    date_format: Optional[str] = "%Y-%m-%d",
    time_format: Optional[str] = "%H:%M:%S",
    datetime_format: Optional[str] = None,   # 若有單欄位 datetime，可指定格式
    use_gpu: bool = False,
    time_stats: bool = False,                # 是否計算昂貴的時間序列特徵
) -> pd.DataFrame:

    df = pd.read_csv(txns_path)
    (from_col, to_col, amt_col,
     date_col, time_col, dt_col,
     channel_col, currency_col,
     from_type, to_type, self_col) = _detect_schema(df)

    # 金額 -> float
    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0).astype("float64")

    # is_self_txn：優先用欄位；否則自己比對 from==to
    if self_col:
        df["is_self_txn"] = _coerce_bool01(df[self_col])
    else:
        df["is_self_txn"] = (df[from_col].astype(str) == df[to_col].astype(str)).astype("int32")

    # 時間戳
    ts = _combine_datetime(df, date_col, time_col, dt_col, date_format, time_format, datetime_format)
    if ts is not None:
        df["_ts_"] = ts
        df["hour"] = df["_ts_"].dt.hour
        df["dow"]  = df["_ts_"].dt.dayofweek
        df["is_night"]   = ((df["hour"] >= 22) | (df["hour"] < 6)).astype("int32")
        df["is_weekend"] = (df["dow"] >= 5).astype("int32")
    else:
        df["is_night"] = 0
        df["is_weekend"] = 0

    # ===== 出金（from）聚合 =====
    if use_gpu and _HAS_CUDF:
        out_block = _group_amt_block_gpu(df, from_col, amt_col, "out")
    else:
        out_block = _group_amt_block_pd(df, from_col, amt_col, "out")

    out_extra = pd.DataFrame({
        "out_night_txn_count":   df.groupby(from_col, sort=False)["is_night"].sum(),
        "out_weekend_txn_count": df.groupby(from_col, sort=False)["is_weekend"].sum(),
        "out_self_txn_count":    df.groupby(from_col, sort=False)["is_self_txn"].sum(),
        "out_unique_to_acct":    df.groupby(from_col, sort=False)[to_col].nunique(),
    })
    if channel_col:
        out_extra["out_channel_nunique"] = df.groupby(from_col, sort=False)[channel_col].nunique()
    if currency_col:
        out_extra["out_currency_nunique"] = df.groupby(from_col, sort=False)[currency_col].nunique()
    if to_type:  # 以對手方型別多樣性當作出金端資訊
        out_extra["out_counterparty_type_nunique"] = df.groupby(from_col, sort=False)[to_type].nunique()

    out_feat = out_block.join(out_extra, how="left")

    # ===== 入金（to）聚合 =====
    if use_gpu and _HAS_CUDF:
        in_block = _group_amt_block_gpu(df, to_col, amt_col, "in")
    else:
        in_block = _group_amt_block_pd(df, to_col, amt_col, "in")

    in_extra = pd.DataFrame({
        "in_night_txn_count":   df.groupby(to_col, sort=False)["is_night"].sum(),
        "in_weekend_txn_count": df.groupby(to_col, sort=False)["is_weekend"].sum(),
        "in_self_txn_count":    df.groupby(to_col, sort=False)["is_self_txn"].sum(),
        "in_unique_from_acct":  df.groupby(to_col, sort=False)[from_col].nunique(),
    })
    if channel_col:
        in_extra["in_channel_nunique"] = df.groupby(to_col, sort=False)[channel_col].nunique()
    if currency_col:
        in_extra["in_currency_nunique"] = df.groupby(to_col, sort=False)[currency_col].nunique()
    if from_type:
        in_extra["in_counterparty_type_nunique"] = df.groupby(to_col, sort=False)[from_type].nunique()

    in_feat = in_block.join(in_extra, how="left")

    # ===== 匯總到帳號 =====
    out_feat.index.name = "acct"
    in_feat.index.name = "acct"
    feat = out_feat.add(in_feat, fill_value=0)

    # ===== 比率特徵 =====
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

    # （可選）時間序列昂貴特徵
    if time_stats and ("_ts_" in df.columns):
        def _burstiness_count(ts: pd.Series, window="5min") -> int:
            s = ts.dropna()
            if s.empty:
                return 0
            s = s.sort_values()
            ones = pd.Series(1, index=s.values)
            return int(ones.rolling(window=window).sum().max())

        def _interarrival_stats(ts: pd.Series) -> tuple[float, float, float]:
            s = ts.dropna()
            if len(s) <= 1:
                return 0.0, 0.0, 0.0
            s = s.sort_values().astype("int64") // 10**9
            diff = np.diff(s.values)
            mean = float(np.mean(diff))
            std  = float(np.std(diff))
            cv   = float(std / (mean + 1e-9))
            return mean, std, cv

        g_out_t = df.groupby(from_col, sort=False)["_ts_"]
        g_in_t  = df.groupby(to_col,   sort=False)["_ts_"]

        out_burst = g_out_t.apply(lambda s: _burstiness_count(s)).rename("out_burst_5min_max")
        in_burst  = g_in_t.apply(lambda s: _burstiness_count(s)).rename("in_burst_5min_max")

        out_iat = g_out_t.apply(_interarrival_stats)
        out_iat = pd.DataFrame(out_iat.tolist(), index=out_iat.index,
                               columns=["out_iat_mean_s", "out_iat_std_s", "out_iat_cv"])
        in_iat = g_in_t.apply(_interarrival_stats)
        in_iat = pd.DataFrame(in_iat.tolist(), index=in_iat.index,
                              columns=["in_iat_mean_s", "in_iat_std_s", "in_iat_cv"])

        feat = feat.join(out_burst, how="left").join(in_burst, how="left").join(out_iat, how="left").join(in_iat, how="left")
        for c in ["out_burst_5min_max", "in_burst_5min_max",
                  "out_iat_mean_s", "out_iat_std_s", "out_iat_cv",
                  "in_iat_mean_s", "in_iat_std_s", "in_iat_cv"]:
            if c in feat.columns:
                feat[c] = feat[c].fillna(0)

    # 數值清理與輸出
    feat = feat.reset_index()
    num_cols = feat.select_dtypes(include=[np.number]).columns
    feat[num_cols] = feat[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ 已輸出特徵：{out_path}  （{feat.shape[0]} rows, {feat.shape[1]} cols）")
    return feat


# ============ CLI ============
def parse_args():
    p = argparse.ArgumentParser(description="Build account-level features from acct_transaction.csv")
    p.add_argument("--txns", default="dataset/acct_transaction.csv", help="交易 CSV 路徑")
    p.add_argument("--out",  default="feature_data/account_features.csv", help="輸出特徵 CSV 路徑")
    p.add_argument("--date-format", default="%Y-%m-%d", help="txn_date 的格式（建議提供以避免警告）")
    p.add_argument("--time-format", default="%H:%M:%S", help="txn_time 的格式")
    p.add_argument("--datetime-format", default=None, help="若使用單欄位 datetime（如 txn_datetime），可指定格式")
    p.add_argument("--gpu", action="store_true", help="若環境有 cuDF，啟用 GPU 加速 groupby")
    p.add_argument("--time-stats", action="store_true", help="計算爆發度/到達間隔（較慢，預設關閉）")
    return p.parse_args()


def main():
    args = parse_args()
    build_account_features(
        txns_path=args.txns,
        out_path=args.out,
        date_format=args.date_format,
        time_format=args.time_format,
        datetime_format=args.datetime_format,
        use_gpu=(args.gpu and _HAS_CUDF),
        time_stats=args.time_stats,
    )


if __name__ == "__main__":
    main()
