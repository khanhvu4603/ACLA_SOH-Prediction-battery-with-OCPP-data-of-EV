# -*- coding: utf-8 -*-
"""
Bước 2 (SOC 20–80%) — HỖ TRỢ NHIỀU FILE DC ĐẦU VÀO
- Join nominal như trước
- Đọc & gộp nhiều file DC_* .csv
- Giữ các phiên phủ đủ 20–80% SOC
- Nội suy lên lưới SOC_GRID = 20..80 (step 1%)
- Tính feature nâng cao (V,I,P,R,soc_norm,delta_soc,dSOC_dt,dV/dSOC,dI/dSOC,dP/dSOC)
- Xuất:
  + ocpp_lstm_dataset.npz  (X: n×61×10, y clip [0.7,1.05])
  + ocpp_lstm_sessions.csv (mỗi phiên 1 dòng, có chỉ số nâng cao)
  + ocpp_lstm_series.csv   (long-form: 61 dòng/phiên)
"""

import os, re
import numpy as np
import pandas as pd

# ===== PATHS =====
TXN_PATH       = r"E:\ACLA + ANODE\data\transaction_fully_202508221207.csv"
TAG_PATH       = r"E:\ACLA + ANODE\data\tbl_user_tag_20250820_15h48.csv"
USER_CAR_PATH  = r"E:\ACLA + ANODE\data\tbl_user_car_model_20250820_14h01.csv"
CAR_TYPE_PATH  = r"E:\ACLA + ANODE\data\car_model_id_with_car_type.csv"
CAR_CAP_PATH   = r"E:\ACLA + ANODE\data\car_type_with_C_bolkWh.csv"

# 👉 LIỆT KÊ NHIỀU FILE DC Ở ĐÂY
DC_INPUT_PATHS = [
    r"E:\ACLA + ANODE\data\DC_thang1_3_2025.csv",
    r"E:\ACLA + ANODE\data\DC_thang9_12_2024.csv",
    r"E:\ACLA + ANODE\data\DC_thang4_2025.xlsx"
    # có thể thêm nhiều file nữa...
]

OUT_DIR        = r"E:\ACLA + ANODE\output_data"
OUT_TXN_NOMINAL= os.path.join(OUT_DIR, "transaction_nominal_capacity.csv")
OUT_DATASET    = os.path.join(OUT_DIR, "ocpp_lstm_dataset.npz")
OUT_META       = os.path.join(OUT_DIR, "ocpp_lstm_sessions.csv")
OUT_SERIES     = os.path.join(OUT_DIR, "ocpp_lstm_series.csv")
OUT_DROPPED    = os.path.join(OUT_DIR, "dropped_txn_no_nominal.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== CONFIG =====
MEAS_KEPT  = ["Voltage", "Current.Import", "Power.Active.Import", "SoC", "Energy.Active.Import.Register"]
SOC_LOW, SOC_HIGH, SOC_STEP = 20.0, 80.0, 1.0
SOC_GRID   = np.arange(SOC_LOW, SOC_HIGH + SOC_STEP, SOC_STEP)  # 61 điểm
MAX_TXN    = None  # debug: giới hạn số phiên (None = tất cả)

# ===== Helpers =====
def parse_kwh_to_float(x) -> float:
    if pd.isna(x): return np.nan
    s = str(x).strip().lower().replace(",", ".")
    m = re.findall(r"[0-9]*\.?[0-9]+", s)
    return float(m[0]) if m else np.nan

def build_transaction_nominal():
    df_txn = pd.read_csv(TXN_PATH);       df_txn.columns = [c.lower() for c in df_txn.columns]
    df_tag = pd.read_csv(TAG_PATH);       df_tag.columns = [c.lower() for c in df_tag.columns]
    df_usr = pd.read_csv(USER_CAR_PATH);  df_usr.columns = [c.lower() for c in df_usr.columns]
    df_typ = pd.read_csv(CAR_TYPE_PATH);  df_typ.columns = [c.lower() for c in df_typ.columns]
    df_cap = pd.read_csv(CAR_CAP_PATH);   df_cap.columns = [c.lower() for c in df_cap.columns]

    df_txn = df_txn[["transaction_pk", "id_tag"]]
    df_tag = df_tag[["tag_id", "user_id"]]
    df_usr = df_usr[["user_id", "car_model_id"]]
    if "car name" in df_typ.columns:
        df_typ = df_typ[["car_model_id", "car name"]].rename(columns={"car name": "car_name"})
    else:
        df_typ = df_typ[["car_model_id", "car_name"]]

    df_cap = df_cap.rename(columns={"c_bol_kwh":"nominal_raw"})
    df_cap["nominal_kwh"] = df_cap["nominal_raw"].map(parse_kwh_to_float)

    m = (df_txn.merge(df_tag, left_on="id_tag", right_on="tag_id", how="left")
               .merge(df_usr, on="user_id", how="left")
               .merge(df_typ, on="car_model_id", how="left")
               .merge(df_cap[["car_name","nominal_kwh"]], on="car_name", how="left"))

    out_all = (m[["transaction_pk","car_name","nominal_kwh"]]
               .dropna(subset=["transaction_pk"]).drop_duplicates().sort_values("transaction_pk"))
    dropped = out_all[out_all["nominal_kwh"].isna() | (out_all["nominal_kwh"] <= 0)]
    kept    = out_all[out_all["nominal_kwh"].notna() & (out_all["nominal_kwh"] > 0)]

    kept.to_csv(OUT_TXN_NOMINAL, index=False)
    dropped.to_csv(OUT_DROPPED, index=False)
    print(f"[JOIN] mapped={len(out_all)} | kept={len(kept)} | dropped(no nominal)={len(dropped)} -> {OUT_TXN_NOMINAL}")
    return kept

def read_and_concat_dc(paths):
    """Đọc nhiều file DC (.csv/.xlsx), chuẩn hoá, gộp, khử trùng."""
    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] DC file not found: {p}")
            continue
        try:
            ext = os.path.splitext(p)[1].lower()
            if ext in [".xlsx", ".xls"]:
                df = pd.read_excel(p)
            else:
                df = pd.read_csv(p)

            df.columns = [c.strip() for c in df.columns]
            if "connector_pk" not in df.columns and "Connector" in df.columns:
                df = df.rename(columns={"Connector": "connector_pk"})

            keep = [c for c in ["transaction_pk","connector_pk","value_timestamp","measurand","value"] if c in df.columns]
            df = df[keep]

            df["value_timestamp"] = pd.to_datetime(df["value_timestamp"], errors="coerce", infer_datetime_format=True)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value_timestamp","value"])
            df = df[df["measurand"].isin(MEAS_KEPT)]
            df["source_file"] = os.path.basename(p)

            # 🔎 LOG: sau khi load từng file
            print(f"[LOAD] {p} -> rows={len(df)}, "
                  f"uniq_txn={df['transaction_pk'].nunique() if 'transaction_pk' in df.columns else 'NA'}")

            frames.append(df)
        except Exception as e:
            print(f"[ERR] reading {p}: {e}")

    if not frames:
        raise RuntimeError("No DC files loaded.")

    big = pd.concat(frames, ignore_index=True)
    # 🔎 LOG: sau khi concat thô
    print("[MERGE raw] total rows:", len(big), "| uniq_txn:", big["transaction_pk"].nunique())

    # Khử trùng lặp theo (txn, ts, measurand)
    if all(c in big.columns for c in ["transaction_pk","value_timestamp","measurand"]):
        big = big.sort_values(["transaction_pk","value_timestamp"])
        big = big.drop_duplicates(subset=["transaction_pk","value_timestamp","measurand"], keep="last")

    big = big.sort_values(["transaction_pk","value_timestamp"])
    # 🔎 LOG: sau khi clean
    print("[MERGE clean] total rows:", len(big), "| uniq_txn:", big["transaction_pk"].nunique())
    return big



def pivot_one_txn(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("value_timestamp")
    pvt = g.pivot_table(index="value_timestamp", columns="measurand", values="value", aggfunc="last")
    pvt = pvt.sort_index().ffill(limit=2)
    for c in MEAS_KEPT:
        if c not in pvt.columns: pvt[c] = np.nan
    return pvt[MEAS_KEPT]

def _ensure_monotonic(x):
    """Ép dãy không giảm (dùng cho SoC)."""
    x = np.asarray(x, dtype=float).copy()
    return np.maximum.accumulate(x)

def _cross_time(t_sec, soc, target):
    """
    Trả về thời điểm (giây) khi SoC đi qua 'target' bằng nội suy tuyến tính trên (t, soc).
    Yêu cầu 'soc' đã đơn điệu tăng.
    """
    if target < soc[0] or target > soc[-1]:
        return None
    idx = np.searchsorted(soc, target)
    if idx == 0 or idx >= len(soc):
        return None
    s0, s1 = soc[idx-1], soc[idx]
    t0, t1 = t_sec[idx-1], t_sec[idx]
    if s1 == s0:
        return t0
    alpha = (target - s0) / (s1 - s0)
    return t0 + alpha * (t1 - t0)

def _interp_time_series(t_src, y_src, t_new):
    """Nội suy y(t) tại các thời điểm t_new (giây)."""
    return np.interp(t_new, t_src, y_src)


def build_features_fixed_grid(pvt: pd.DataFrame):
    """
    Ép đúng 20–80 bằng nội suy theo thời gian:
      - Tìm t20, t80 (khi SOC=20,80) trên (t, SoC) đã ép đơn điệu.
      - Cắt chuỗi theo [t20, t80], chèn biên bằng nội suy V/I/P.
      - Tính E_kWh bằng trapezoid trên P(t) → usable_kWh = E/0.60.
      - Nội suy V/I/P lên SOC_GRID (20..80), tính R, dV/dSOC, dI/dSOC, dP/dSOC.
      - Tính dSOC/dt trên lưới SOC qua t(SOC) rồi vi phân.
    Trả về: X_seq (61×10), usable_kwh (float), extras (dict).
    """
    # 0) kiểm tra dữ liệu
    if "SoC" not in pvt.columns or pvt["SoC"].dropna().empty:
        return None
    pvt = pvt.sort_index()
    cols = ["SoC","Voltage","Current.Import","Power.Active.Import"]
    q = pvt[cols].dropna(subset=["SoC"]).copy()
    q[["Voltage","Current.Import","Power.Active.Import"]] = q[["Voltage","Current.Import","Power.Active.Import"]].ffill(limit=2)
    q = q.dropna()
    if q.empty:
        return None

    # 1) thời gian & SOC (đơn điệu)
    t_idx = q.index
    t_sec = (t_idx.view("int64") / 1e9).astype(float)  # giây
    soc_raw = q["SoC"].values.astype(float)
    soc_mono = _ensure_monotonic(soc_raw)

    # cần phủ qua 20..80
    if soc_mono.min() > SOC_LOW or soc_mono.max() < SOC_HIGH:
        return None

    # 2) t20, t80
    t20 = _cross_time(t_sec, soc_mono, SOC_LOW)
    t80 = _cross_time(t_sec, soc_mono, SOC_HIGH)
    if t20 is None or t80 is None or t80 <= t20:
        return None

    # 3) cắt theo thời gian, chèn biên
    P_t = q["Power.Active.Import"].values
    V_t = q["Voltage"].values
    I_t = q["Current.Import"].values
    mask = (t_sec >= t20) & (t_sec <= t80)
    t_clip = t_sec[mask]
    P_clip = P_t[mask]
    V_clip = V_t[mask]
    I_clip = I_t[mask]
    # thêm điểm biên nếu thiếu
    if len(t_clip) == 0 or t_clip[0] > t20 + 1e-9:
        t_clip = np.insert(t_clip, 0, t20)
        P_clip = np.insert(P_clip, 0, _interp_time_series(t_sec, P_t, t20))
        V_clip = np.insert(V_clip, 0, _interp_time_series(t_sec, V_t, t20))
        I_clip = np.insert(I_clip, 0, _interp_time_series(t_sec, I_t, t20))
    if t_clip[-1] < t80 - 1e-9:
        t_clip = np.append(t_clip, t80)
        P_clip = np.append(P_clip, _interp_time_series(t_sec, P_t, t80))
        V_clip = np.append(V_clip, _interp_time_series(t_sec, V_t, t80))
        I_clip = np.append(I_clip, _interp_time_series(t_sec, I_t, t80))

    # 4) E_kWh chính xác trong [t20,t80] bằng trapezoid
    dt_h = np.diff(t_clip) / 3600.0
    P_kw_mid = 0.5 * (P_clip[:-1] + P_clip[1:]) / 1000.0
    E_kWh = np.sum(P_kw_mid * dt_h)
    usable_kwh = E_kWh / 0.60 if E_kWh > 0 else np.nan

    # 5) Nội suy V/I/P theo SOC lên SOC_GRID
    df_soc = pd.DataFrame({"soc": soc_mono, "V": V_t, "I": I_t, "P": P_t}).dropna()
    df_soc = df_soc.drop_duplicates(subset="soc").sort_values("soc")
    if df_soc.shape[0] < 10:
        return None
    V = np.interp(SOC_GRID, df_soc["soc"].values, df_soc["V"].values)
    I = np.interp(SOC_GRID, df_soc["soc"].values, df_soc["I"].values)
    P = np.interp(SOC_GRID, df_soc["soc"].values, df_soc["P"].values)

    # 6) R và các đạo hàm
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(I != 0, V / I, np.nan)
    if np.any(~np.isfinite(R)):
        good = np.isfinite(R)
        R[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), R[good]) if good.sum()>=2 else np.nanmean(R)
    dV_dSOC = np.gradient(V, SOC_GRID)
    dI_dSOC = np.gradient(I, SOC_GRID)
    dP_dSOC = np.gradient(P, SOC_GRID)

    # 7) dSOC/dt trên lưới SOC
    t_of_soc = np.interp(SOC_GRID, soc_mono, t_sec)  # giây
    dt_sec = np.gradient(t_of_soc, SOC_GRID)
    with np.errstate(divide="ignore", invalid="ignore"):
        dSOC_dt_seq = np.where(dt_sec > 0, 1.0 / dt_sec, 0.0)
    dSOC_dt_mean = float(np.nanmean(dSOC_dt_seq))
    dSOC_dt_max  = float(np.nanmax(dSOC_dt_seq))

    # 8) Gộp features (delta_soc_20_80 ép 60)
    soc_norm = (SOC_GRID - SOC_LOW) / (SOC_HIGH - SOC_LOW)
    delta_soc_seq = np.full_like(SOC_GRID, 60.0, dtype=float)
    X_seq = np.stack([V, I, P, R, soc_norm, delta_soc_seq, dSOC_dt_seq, dV_dSOC, dI_dSOC, dP_dSOC], axis=1)

    extras = {
        "soc_min_full": float(soc_mono.min()),
        "soc_max_full": float(soc_mono.max()),
        "delta_soc_full": float(soc_mono.max() - soc_mono.min()),
        "soc_min_20_80": 20.0,
        "soc_max_20_80": 80.0,
        "delta_soc_20_80": 60.0,
        "dSOC_dt_mean": dSOC_dt_mean,
        "dSOC_dt_max": dSOC_dt_max
    }
    return X_seq, usable_kwh, extras

def preprocess_and_save(txn_nominal_df: pd.DataFrame):
    # 1) Đọc & gộp nhiều file DC
    big = read_and_concat_dc(DC_INPUT_PATHS)

    # 2) Lọc theo nominal có match
    matched = set(txn_nominal_df["transaction_pk"].unique().tolist())
    before = big["transaction_pk"].nunique()
    df = big[big["transaction_pk"].isin(matched)]
    after  = df["transaction_pk"].nunique()

    # 🔎 LOG: sau khi lọc theo mapping nominal
    print(f"[FILTER nominal] txn before={before} -> after={after} "
          f"(dropped {before-after} txn not in nominal map)")

    X_list, y_list, meta_rows, series_rows = [], [], [], []

    txn_ids = sorted(df["transaction_pk"].unique())
    if isinstance(MAX_TXN, int):
        txn_ids = txn_ids[:MAX_TXN]

    # 🔢 Bộ đếm lý do rớt
    cnt_ok = 0
    cnt_soc_cover = 0
    cnt_other = 0

    for txn_id in txn_ids:
        g = df[df["transaction_pk"] == txn_id]
        pvt = pivot_one_txn(g)
        built = build_features_fixed_grid(pvt)
        if built is None:
            # phân loại lý do
            soc = pvt["SoC"].dropna()
            if soc.empty or soc.min() > SOC_LOW or soc.max() < SOC_HIGH:
                cnt_soc_cover += 1  # rớt vì không phủ 20–80
            else:
                cnt_other += 1      # rớt vì lý do khác (thiếu cột, <10 điểm SOC, v.v.)
            continue

        cnt_ok += 1
        X_seq, usable_kwh, extras = built

        row = txn_nominal_df[txn_nominal_df["transaction_pk"] == txn_id]
        nominal = float(row["nominal_kwh"].values[0])
        car_name = row["car_name"].values[0]

        if pd.isna(usable_kwh) or nominal <= 0:
            cnt_other += 1
            continue

        soh_abs = usable_kwh / nominal
        X_list.append(X_seq)
        y_list.append(soh_abs)

        meta_rows.append({
            "txn_id": txn_id,
            "car_name": car_name,
            "nominal_kwh": nominal,
            "usable_kwh": usable_kwh,
            "label_soh_abs": soh_abs,
            "label_soh": float(np.clip(soh_abs, 0.7, 1.05)),
            # FULL session
            "soc_min_full": extras["soc_min_full"],
            "soc_max_full": extras["soc_max_full"],
            "delta_soc_full": extras["delta_soc_full"],
            # 20–80 window (đúng với feature/nhãn)
            "soc_min_20_80": extras["soc_min_20_80"],
            "soc_max_20_80": extras["soc_max_20_80"],
            "delta_soc_20_80": extras["delta_soc_20_80"],
            "dSOC_dt_mean": extras["dSOC_dt_mean"],
            "dSOC_dt_max": extras["dSOC_dt_max"],
            "start_ts": g["value_timestamp"].min(),
            "end_ts": g["value_timestamp"].max(),
            "connector": g["connector_pk"].iloc[0] if "connector_pk" in g.columns else np.nan,
            "seq_len": X_seq.shape[0]  # 61
        })


        for i, soc in enumerate(SOC_GRID):
            series_rows.append({
                "txn_id": txn_id,
                "soc": float(soc),
                "V": float(X_seq[i,0]),
                "I": float(X_seq[i,1]),
                "P": float(X_seq[i,2]),
                "R": float(X_seq[i,3]),
                "soc_norm": float(X_seq[i,4]),
                "delta_soc_20_80": float(extras["delta_soc_20_80"]),
                "dSOC_dt": float(X_seq[i,6]),
                "dV_dSOC": float(X_seq[i,7]),
                "dI_dSOC": float(X_seq[i,8]),
                "dP_dSOC": float(X_seq[i,9]),
            })

     # 🔎 LOG: tổng kết lý do rớt
    print(f"[SUMMARY] OK={cnt_ok}, drop_no_20_80={cnt_soc_cover}, drop_other={cnt_other}")

    if not X_list:
        raise RuntimeError("Không có phiên hợp lệ sau khi lọc nominal & phủ đủ SOC 20–80% từ các file DC.")

    X = np.stack(X_list, axis=0)                  # (n, 61, 10)
    y = np.asarray(y_list, dtype=float)
    y_clipped = np.clip(y, 0.7, 1.05)

    features = ["V","I","P","R","soc_norm","delta_soc","dSOC_dt","dV_dSOC","dI_dSOC","dP_dSOC"]
    np.savez(OUT_DATASET, X=X, y=y_clipped, soc_grid=SOC_GRID, features=features)
    print(f"[PRE] saved dataset: {OUT_DATASET}  | X={X.shape}, y={y_clipped.shape}")

    meta_df = pd.DataFrame(meta_rows); meta_df.to_csv(OUT_META, index=False)
    print(f"[PRE] saved meta  : {OUT_META}  ({len(meta_df)} phiên)")

    series_df = pd.DataFrame(series_rows); series_df.to_csv(OUT_SERIES, index=False)
    print(f"[PRE] saved series: {OUT_SERIES}  ({len(series_df)} dòng)")

    return X, y_clipped, meta_df, series_df

# ===== MAIN =====
if __name__ == "__main__":
    txn_nominal_df = build_transaction_nominal()
    X, y, meta_df, series_df = preprocess_and_save(txn_nominal_df)
    print("Done.")
