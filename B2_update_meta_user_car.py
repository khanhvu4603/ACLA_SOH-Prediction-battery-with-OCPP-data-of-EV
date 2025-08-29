#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1_update_meta_user_car.py
---------------------------------
Bổ sung cột user_id / car_id (và thông tin xe) vào meta `ocpp_lstm_sessions.csv`
từ các bảng OCPP/raw bạn đã có. Script cố gắng dò cột & join "best-effort".

CÁCH DÙNG
---------
1) Sửa phần CONFIG phía dưới cho đúng đường dẫn file của bạn
2) Chạy:
   python B1_update_meta_user_car.py

KẾT QUẢ
-------
- Ghi file mới: ocpp_lstm_sessions_with_user_car.csv (giữ nguyên thứ tự hàng)
- Các cột thêm (nếu tìm thấy): 
  user_id, tag_id, car_id, car_model_id, car_name, car_type, C_bolkWh

YÊU CẦU
-------
pip install pandas numpy
"""

# ============ CONFIG ============
# META_PATH = r"./output_data/ocpp_lstm_sessions.csv"              # meta do B1_prepocessing.py sinh
# TRANSACTIONS_CSV = r"./data/transaction_fully_202508221207.csv"  # bảng transaction đầy đủ
# USER_TAG_CSV     = r"./data/tbl_user_tag_20250820_15h48.csv"      # map tag_id -> user_id (nếu có)
# USER_CAR_MODEL   = r"./data/tbl_user_car_model_20250820_14h01.csv" # map user_id -> car_id|car_model_id (nếu có)
# CAR_MODEL_MAP    = r"./data/car_model_id_with_car_type.csv"       # map car_model_id -> car_name|car_type (nếu có)
# CAR_TYPE_CAP     = r"./data/car_type_with_C_bolkWh.csv"           # map car_type -> C_bolkWh (nếu có)

# OUTPUT_PATH      = r"./output_data/ocpp_lstm_sessions_with_user_car.csv"

META_PATH = r"E:\ACLA + ANODE\output_infer\ocpp_lstm_infer_sessions.csv"              # meta do B1_prepocessing.py sinh
TRANSACTIONS_CSV = r"./data/transaction_fully_202508221207.csv"  # bảng transaction đầy đủ
USER_TAG_CSV     = r"./data/tbl_user_tag_20250820_15h48.csv"      # map tag_id -> user_id (nếu có)
USER_CAR_MODEL   = r"./data/tbl_user_car_model_20250820_14h01.csv" # map user_id -> car_id|car_model_id (nếu có)
CAR_MODEL_MAP    = r"./data/car_model_id_with_car_type.csv"       # map car_model_id -> car_name|car_type (nếu có)
CAR_TYPE_CAP     = r"./data/car_type_with_C_bolkWh.csv"           # map car_type -> C_bolkWh (nếu có)

OUTPUT_PATH      = r"./output_infer/ocpp_lstm_sessions_with_user_car.csv"
# =================================

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def read_csv_safe(path):
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Không tìm thấy file: {p}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[ERROR] Lỗi đọc {p}: {e}")
        return None

def _lower_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def main():
    # --- Load & chuẩn hoá tên cột (lower) ---
    meta = read_csv_safe(META_PATH)
    if meta is None:
        sys.exit(1)
    meta = _lower_cols(meta)
    meta["__row_idx__"] = np.arange(len(meta))

    trx  = read_csv_safe(TRANSACTIONS_CSV)
    utag = read_csv_safe(USER_TAG_CSV)
    ucar = read_csv_safe(USER_CAR_MODEL)
    cmap = read_csv_safe(CAR_MODEL_MAP)
    ccap = read_csv_safe(CAR_TYPE_CAP)

    if trx is not None:  trx  = _lower_cols(trx)
    if utag is not None: utag = _lower_cols(utag)
    if ucar is not None: ucar = _lower_cols(ucar)
    if cmap is not None: cmap = _lower_cols(cmap)
    if ccap is not None: ccap = _lower_cols(ccap)

    # =======================
    # 1) meta ↔ transactions: dùng transaction_pk
    # =======================
    # Tìm khóa ở meta: ưu tiên 'transaction_pk' (nếu meta có), fallback: 'transaction_id'/'session_id'.
    key_meta_candidates = ["transaction_pk", "transaction_id", "session_id", "txn_id", "id"]
    key_meta = next((c for c in key_meta_candidates if c in meta.columns), None)

    if trx is not None:
        # Lấy 'transaction_pk' và 'tag_id' (hoặc 'id_tag')
        trx_key = "transaction_pk" if "transaction_pk" in trx.columns else None
        trx_tag = "tag_id" if "tag_id" in trx.columns else ("id_tag" if "id_tag" in trx.columns else None)

        if trx_key is None:
            print("[WARN] TRANSACTIONS_CSV không có 'transaction_pk' → không thể join theo yêu cầu.")
        if trx_tag is None:
            print("[WARN] TRANSACTIONS_CSV không có 'tag_id'/'id_tag' → không thể lấy tag.")

        if trx_key and trx_tag and key_meta:
            if key_meta != trx_key:
                # Nếu meta không có transaction_pk, vẫn thử join dùng key_meta ↔ trx_key nếu trùng nghĩa
                print(f"[JOIN] meta.{key_meta}  <->  trx.{trx_key}  (để lấy {trx_tag})")
                left_on, right_on = key_meta, trx_key
            else:
                print(f"[JOIN] meta.transaction_pk  <->  trx.transaction_pk  (để lấy {trx_tag})")
                left_on, right_on = "transaction_pk", "transaction_pk"

            meta = meta.merge(
                trx[[trx_key, trx_tag]].drop_duplicates(),
                left_on=left_on, right_on=right_on, how="left"
            )
            # Đặt tên cột tag_id nhất quán
            if trx_tag != "tag_id":
                meta.rename(columns={trx_tag: "tag_id"}, inplace=True)
        else:
            print("[WARN] Bỏ qua bước join transactions (thiếu cột bắt buộc).")

    # =======================
    # 2) tag_id → user_id (tbl_user_tag)
    # =======================
    if utag is not None:
        # Chuẩn cột dự kiến: tag_id / user_id
        # Chấp nhận alias id_tag → tag_id
        if "id_tag" in utag.columns and "tag_id" not in utag.columns:
            utag.rename(columns={"id_tag": "tag_id"}, inplace=True)
        if "id_user" in utag.columns and "user_id" not in utag.columns:
            utag.rename(columns={"id_user": "user_id"}, inplace=True)

        if "tag_id" in utag.columns and "user_id" in utag.columns and "tag_id" in meta.columns:
            print("[JOIN] tag_id → user_id (tbl_user_tag)")
            meta = meta.merge(
                utag[["tag_id", "user_id"]].drop_duplicates(),
                on="tag_id", how="left"
            )
        else:
            print("[WARN] Không thể ánh xạ tag_id→user_id (thiếu 'tag_id' hoặc 'user_id').")
    else:
        print("[WARN] Không có USER_TAG_CSV, bỏ qua map tag→user.")

    # =======================
    # 3) user_id → car_model_id (tbl_user_car_model)
    # =======================
    if ucar is not None:
        # Chuẩn tên
        if "id_user" in ucar.columns and "user_id" not in ucar.columns:
            ucar.rename(columns={"id_user": "user_id"}, inplace=True)
        if "model_id" in ucar.columns and "car_model_id" not in ucar.columns:
            ucar.rename(columns={"model_id": "car_model_id"}, inplace=True)

        if "user_id" in meta.columns and "user_id" in ucar.columns and "car_model_id" in ucar.columns:
            print("[JOIN] user_id → car_model_id (tbl_user_car_model)")
            meta = meta.merge(
                ucar[["user_id", "car_model_id"]].drop_duplicates(),
                on="user_id", how="left"
            )
        else:
            print("[WARN] Không thể ánh xạ user_id→car_model_id (thiếu cột).")
    else:
        print("[WARN] Không có USER_CAR_MODEL, bỏ qua map user→car_model.")

    # =======================
    # 4) car_model_id → car_name / car_type
    # =======================
    if cmap is not None:
        # Chuẩn tên
        if "model_id" in cmap.columns and "car_model_id" not in cmap.columns:
            cmap.rename(columns={"model_id": "car_model_id"}, inplace=True)
        if "model_name" in cmap.columns and "car_name" not in cmap.columns:
            cmap.rename(columns={"model_name": "car_name"}, inplace=True)
        if "type" in cmap.columns and "car_type" not in cmap.columns:
            cmap.rename(columns={"type": "car_type"}, inplace=True)

        if "car_model_id" in meta.columns and "car_model_id" in cmap.columns:
            keep = ["car_model_id"]
            if "car_name" in cmap.columns: keep.append("car_name")
            if "car_type" in cmap.columns: keep.append("car_type")
            print("[JOIN] car_model_id → car_name / car_type")
            meta = meta.merge(cmap[keep].drop_duplicates(), on="car_model_id", how="left")
        else:
            print("[WARN] Không thể ánh xạ car_model_id→car_name/car_type (thiếu car_model_id).")
    else:
        print("[WARN] Không có CAR_MODEL_MAP, bỏ qua map model→name/type.")

    # =======================
    # 5) car_type → C_bolkWh
    # =======================
    if ccap is not None:
        # Chuẩn tên
        if "type" in ccap.columns and "car_type" not in ccap.columns:
            ccap.rename(columns={"type": "car_type"}, inplace=True)
        if "capacity_kwh" in ccap.columns and "c_bolkwh" not in ccap.columns:
            ccap.rename(columns={"capacity_kwh": "c_bolkwh"}, inplace=True)
        if "nominal_kwh" in ccap.columns and "c_bolkwh" not in ccap.columns:
            ccap.rename(columns={"nominal_kwh": "c_bolkwh"}, inplace=True)

        if "car_type" in meta.columns and "car_type" in ccap.columns and "c_bolkwh" in ccap.columns:
            print("[JOIN] car_type → C_bolkWh")
            meta = meta.merge(ccap[["car_type","c_bolkwh"]].drop_duplicates(), on="car_type", how="left")
        else:
            print("[WARN] Không thể ánh xạ car_type→C_bolkWh (thiếu cột).")
    else:
        print("[WARN] Không có CAR_TYPE_CAP, bỏ qua map type→capacity.")

    # ===== Hoàn tất: sắp xếp theo thứ tự ban đầu, đặt lại thứ tự cột ưu tiên =====
    meta = meta.sort_values("__row_idx__").drop(columns=["__row_idx__"])

    preferred = [c for c in ["transaction_pk","transaction_id","session_id","tag_id","user_id",
                             "car_model_id","car_name","car_type","c_bolkwh"] if c in meta.columns]
    other = [c for c in meta.columns if c not in preferred]
    meta = meta[preferred + other]

    out = Path(OUTPUT_PATH)
    meta.to_csv(out, index=False, encoding="utf-8")
    print(f"[DONE] Ghi meta đã bổ sung user/xe -> {out.resolve()}")
    print(f"[INFO] Số dòng: {len(meta)} | Cột: {len(meta.columns)}")

if __name__ == "__main__":
    main()
