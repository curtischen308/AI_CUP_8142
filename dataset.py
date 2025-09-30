

import os
import pandas as pd
import unicodedata

def _clean_str_series(s: pd.Series) -> pd.Series:
    # 將 NA -> ""，標準化、去不可見字元、前後空白、轉小寫
    s = s.astype("string").fillna("")
    s = s.map(lambda x: unicodedata.normalize("NFKC", x))
    # 去除常見不可見/空白
    s = s.str.replace(r"[\u200B-\u200D\uFEFF]", "", regex=True)  # 零寬/FEFF
    s = s.str.strip().str.lower()
    return s

def split_by_alert(
    trans_path: str,
    alert_path: str,
    out_dir: str = ".",
    chunk_size: int = 200_000,
):
    os.makedirs(out_dir, exist_ok=True)

    # === 讀取警示帳戶 ===
    try:
        alert_df = pd.read_csv(alert_path, encoding="utf-8-sig", dtype="string")
    except UnicodeDecodeError:
        # 若不是 UTF-8，可再試常見編碼
        alert_df = pd.read_csv(alert_path, encoding="utf-8", dtype="string")

    # 若沒有 acct 欄，嘗試把第一欄視為 acct
    if "acct" not in alert_df.columns:
        first_col = alert_df.columns[0]
        alert_df = alert_df.rename(columns={first_col: "acct"})

    alert_df["acct"] = _clean_str_series(alert_df["acct"])
    alert_accts = set(alert_df["acct"][alert_df["acct"] != ""])

    # 診斷：警示帳戶數
    print(f"[INFO] 警示帳戶數量：{len(alert_accts)}")
    if len(alert_accts) == 0:
        print("[WARNING] 警示帳戶清單為空，請檢查 acct_alert.csv 的欄名/編碼/內容。")

    # === 準備輸出 ===
    out_both = os.path.join(out_dir, "txn_from_to_both_in_alert.csv")  #轉出帳號與轉入帳號都在警示清單裡 
    out_from = os.path.join(out_dir, "txn_from_only_in_alert.csv")     #只有轉出帳號在警示清單
    out_to   = os.path.join(out_dir, "txn_to_only_in_alert.csv")       #只有轉入帳號在警示清單
    out_none = os.path.join(out_dir, "txn_neither_in_alert.csv")       #轉出、轉入帳號都不在警示清單

    for p in [out_both, out_from, out_to, out_none]:
        if os.path.exists(p):
            os.remove(p)

    # === 讀取交易（分塊）===
    use_dtypes = {
        "from_acct": "string",
        "from_acct_type": "string",
        "to_acct": "string",
        "to_acct_type": "string",
        "is_self_txn": "string",
        "txn_amt": "float64",
        "txn_date": "string",
        "txn_time": "string",
        "currency_type": "string",
        "channel_type": "string",
    }

    reader = pd.read_csv(
        trans_path,
        dtype=use_dtypes,
        chunksize=chunk_size,
        low_memory=True,
        encoding="utf-8-sig"  # 若原檔非 UTF-8-SIG 也能正常讀
    )

    first_write = {out_both: True, out_from: True, out_to: True, out_none: True}

    # 累計計數以便驗證切分總數是否等於原始總數
    total_rows = 0
    cnt_both = cnt_from = cnt_to = cnt_none = 0

    for i, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        # 清理帳號欄位
        chunk["from_acct"] = _clean_str_series(chunk["from_acct"])
        chunk["to_acct"]   = _clean_str_series(chunk["to_acct"])

        from_in = chunk["from_acct"].isin(alert_accts)
        to_in   = chunk["to_acct"].isin(alert_accts)

        df_both = chunk[from_in & to_in]
        df_from = chunk[from_in & ~to_in]
        df_to   = chunk[~from_in & to_in]
        df_none = chunk[~from_in & ~to_in]

        cnt_both += len(df_both)
        cnt_from += len(df_from)
        cnt_to   += len(df_to)
        cnt_none += len(df_none)

        # 寫檔
        for df, path in [(df_both, out_both), (df_from, out_from), (df_to, out_to), (df_none, out_none)]:
            if df.empty:
                continue
            df.to_csv(path, index=False, mode="w" if first_write[path] else "a", header=first_write[path])
            first_write[path] = False

        # 每批簡短診斷
        print(f"[CHUNK {i}] both={len(df_both)}, from_only={len(df_from)}, to_only={len(df_to)}, neither={len(df_none)}")

    print("\n[SUMMARY]")
    print(f"total_rows = {total_rows}")
    print(f"both       = {cnt_both}")
    print(f"from_only  = {cnt_from}")
    print(f"to_only    = {cnt_to}")
    print(f"neither    = {cnt_none}")
    print(f"check sum  = {cnt_both + cnt_from + cnt_to + cnt_none} (應等於 total_rows)")

if __name__ == "__main__":
    split_by_alert(
        trans_path="./dataset/acct_transaction.csv",
        alert_path="./dataset/acct_alert.csv",
        out_dir="./sort_data",
        chunk_size=200_000,
    )