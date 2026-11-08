"""
CareLoop Clinics – No-Show Guard
PART 1: COMPLETE PYTHON PIPELINE (Tasks 1.1–1.6)

Input files (place beside this script):
  qDCCmpyfss1kBJJB.csv
  clinics.csv

Install:
  pip install pandas numpy requests

Run:
  python task1_complete.py

Generated files are saved in the outputs/ folder.
Open-Meteo weather API does not require an API key.
"""

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

APPOINTMENTS_FILE = BASE_DIR / "qDCCmpyfss1kBJJB.csv"
CLINICS_FILE = BASE_DIR / "clinics.csv"

REJECTED_FILE = OUTPUT_DIR / "rejected_rows.json"
PASS1_FILE = OUTPUT_DIR / "appointments_pass1.csv"
PASS2_FILE = OUTPUT_DIR / "appointments_pass2.csv"
FEATURES_FILE = OUTPUT_DIR / "appointments_features.csv"
CLINICS_OUT_FILE = OUTPUT_DIR / "clinics_enriched.csv"
FINAL_FILE = OUTPUT_DIR / "appointments_clean.csv"
SQL_FILE = OUTPUT_DIR / "load_data.sql"

FINAL_COLUMNS = [
    "appointment_id", "patient_id", "clinic_id", "booked_on", "appt_date",
    "age", "gender", "distance_km", "fee_inr", "booking_channel",
    "visit_type", "reminder_sent", "prior_no_shows", "status",
    "lead_days", "no_show"
]


def find_col(df, choices, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for choice in choices:
        if choice.lower() in lookup:
            return lookup[choice.lower()]
    if required:
        raise KeyError(f"Expected one of {choices}; found columns: {list(df.columns)}")
    return None


def sql_value(value):
    if pd.isna(value):
        return "NULL"
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "NULL" if not math.isfinite(float(value)) else repr(float(value))
    return "'" + str(value).replace("'", "''") + "'"


# TASK 1.1 – Read the raw CSV with Python's built-in csv module and validate rows.
def task_1_1():
    print("\nTASK 1.1 – CSV INGESTION AND VALIDATION")
    if not APPOINTMENTS_FILE.exists():
        raise FileNotFoundError(f"Upload {APPOINTMENTS_FILE.name} beside this script.")

    accepted, rejected = [], []
    with APPOINTMENTS_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            try:
                clean = dict(row)
                for col in ["age", "fee_inr", "prior_no_shows"]:
                    if col in clean and str(clean[col]).strip():
                        value = float(clean[col])
                        if not value.is_integer():
                            raise ValueError(f"{col} must be an integer")
                        clean[col] = int(value)
                if "distance_km" in clean and str(clean["distance_km"]).strip():
                    clean["distance_km"] = float(clean["distance_km"])
                accepted.append(clean)
            except Exception as exc:
                rejected.append({
                    "row_number": row_number,
                    "row": row,
                    "reason": str(exc)
                })

    REJECTED_FILE.write_text(
        json.dumps(rejected, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    print("Accepted rows:", len(accepted))
    print("Rejected rows:", len(rejected))
    print("Rejected-row report:", REJECTED_FILE)
    return pd.DataFrame(accepted)


# TASK 1.2 – Inspect data, remove duplicates, standardise categories and parse dates.
def task_1_2(df):
    print("\nTASK 1.2 – PANDAS INSPECTION AND CLEANING")
    print("\nHEAD:\n", df.head())
    print("\nTAIL:\n", df.tail())
    print("\nSHAPE:", df.shape)
    print("\nINFO:")
    df.info()
    print("\nDESCRIBE:\n", df.describe(include="all"))
    print("\nMISSING VALUES:\n", df.isna().sum())
    print("\nEXACT DUPLICATES:", int(df.duplicated().sum()))

    df = df.drop_duplicates().copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    for col in ["gender", "booking_channel", "visit_type", "reminder_sent", "status"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.lower()

    for col in ["booked_on", "appt_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    required = [c for c in ["appt_date", "status"] if c in df.columns]
    before = len(df)
    if required:
        df = df.dropna(subset=required).copy()

    print("Rows removed for missing/invalid appointment date or status:", before - len(df))
    print("Rows after Pass 1:", len(df))
    print("\nMISSING VALUES AFTER CLEANING:\n", df.isna().sum())
    df.to_csv(PASS1_FILE, index=False)
    print("Saved:", PASS1_FILE)
    return df


# TASK 1.3 – Fix invalid values, fill numeric gaps and cap IQR outliers.
def task_1_3(df):
    print("\nTASK 1.3 – VALIDITY, MISSING VALUES AND OUTLIERS")
    for col in ["age", "distance_km", "fee_inr", "prior_no_shows"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "age" in df.columns:
        invalid = (df["age"] < 1) | (df["age"] > 100)
        print("Invalid ages:", int(invalid.sum()))
        df.loc[invalid, "age"] = np.nan
        if df["age"].notna().any():
            df["age"] = df["age"].fillna(df["age"].median())

    for col in ["distance_km", "fee_inr", "prior_no_shows"]:
        if col in df.columns and df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in ["distance_km", "fee_inr"]:
        if col in df.columns and df[col].notna().any():
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            count = int(((df[col] < lower) | (df[col] > upper)).sum())
            df[col] = df[col].clip(lower, upper)
            print(
                f"{col}: Q1={q1:.2f}, Q3={q3:.2f}, "
                f"lower={lower:.2f}, upper={upper:.2f}, capped={count}"
            )

    if "distance_km" in df.columns:
        df["distance_km"] = df["distance_km"].round(1)
    for col in ["fee_inr", "age", "prior_no_shows"]:
        if col in df.columns:
            df[col] = df[col].round(0).astype("Int64")

    print("\nMISSING VALUES AFTER PASS 2:\n", df.isna().sum())
    print("\nNUMERIC SUMMARY:\n", df.select_dtypes(include=np.number).describe())
    df.to_csv(PASS2_FILE, index=False)
    print("Saved:", PASS2_FILE)
    return df


# TASK 1.4 – Create lead_days and no_show; use NumPy for calculations.
def task_1_4(df):
    print("\nTASK 1.4 – FEATURE ENGINEERING AND NUMPY")
    df = df.copy()
    df["booked_on"] = pd.to_datetime(df["booked_on"], errors="coerce")
    df["appt_date"] = pd.to_datetime(df["appt_date"], errors="coerce")
    df["lead_days"] = (df["appt_date"] - df["booked_on"]).dt.days

    invalid_lead = df["lead_days"].isna() | (df["lead_days"] < 0)
    print("Invalid lead-time rows removed:", int(invalid_lead.sum()))
    df = df.loc[~invalid_lead].copy()

    status = df["status"].astype(str).str.strip().str.lower()
    df["no_show"] = status.isin(["no_show", "no-show", "noshow", "no show"]).astype(int)

    numeric_cols = ["age", "distance_km", "fee_inr", "lead_days"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    matrix = df[numeric_cols].to_numpy(dtype=float)
    means = np.mean(matrix, axis=0)
    print("\nNumPy column means:")
    for col, mean in zip(numeric_cols, means):
        print(f"{col}: {mean:.2f}")

    long_lead = int((df["lead_days"] > 14).sum())
    print(
        f"\nLead time >14 days: {long_lead} "
        f"({long_lead / len(df) * 100:.1f}%)"
    )

    distances = df["distance_km"].to_numpy(dtype=float)
    if len(distances) > 1 and np.std(distances) > 0:
        z = (distances - np.mean(distances)) / np.std(distances)
        print("Distance observations with |z-score| > 2:", int(np.sum(np.abs(z) > 2)))
    else:
        print("Distance observations with |z-score| > 2: 0")

    print("No-show count:", int(df["no_show"].sum()))
    print("No-show percentage:", round(df["no_show"].mean() * 100, 2), "%")
    df.to_csv(FEATURES_FILE, index=False)
    print("Saved:", FEATURES_FILE)
    return df


# TASK 1.5 – Fetch clinic temperature from Open-Meteo (no API key required).
def task_1_5():
    print("\nTASK 1.5 – CLINIC WEATHER ENRICHMENT")
    if not CLINICS_FILE.exists():
        raise FileNotFoundError(f"Upload {CLINICS_FILE.name} beside this script.")

    clinics = pd.read_csv(CLINICS_FILE)
    lat_col = find_col(clinics, ["latitude", "lat"])
    lon_col = find_col(clinics, ["longitude", "lon", "lng"])
    clinics["temp_c"] = np.nan
    endpoint = "https://api.open-meteo.com/v1/forecast"

    for idx, row in clinics.iterrows():
        if pd.isna(row[lat_col]) or pd.isna(row[lon_col]):
            print(f"Clinic row {idx}: coordinates missing; skipped")
            continue

        params = {
            "latitude": float(row[lat_col]),
            "longitude": float(row[lon_col]),
            "hourly": "temperature_2m",
            "forecast_days": 1,
            "timezone": "auto",
        }

        for attempt in range(2):
            try:
                response = requests.get(endpoint, params=params, timeout=20)
                response.raise_for_status()
                temperatures = response.json().get("hourly", {}).get("temperature_2m", [])
                temperatures = [x for x in temperatures if x is not None]
                if not temperatures:
                    raise ValueError("No temperature data returned")
                clinics.loc[idx, "temp_c"] = float(temperatures[0])
                print(f"Clinic row {idx}: {temperatures[0]} °C")
                break
            except Exception as exc:
                print(f"Clinic row {idx}, attempt {attempt + 1}: {exc}")
                if attempt == 0:
                    time.sleep(2)
        time.sleep(0.2)

    clinics.to_csv(CLINICS_OUT_FILE, index=False)
    print("Clinic records:", len(clinics))
    print("Temperature values collected:", int(clinics["temp_c"].notna().sum()))
    print("Saved:", CLINICS_OUT_FILE)
    return clinics


# TASK 1.6 – Export exact final appointment columns and SQL INSERT statements.
def task_1_6(df, clinics):
    print("\nTASK 1.6 – FINAL EXPORT AND SQL SCRIPT")
    missing = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing final appointment columns: {missing}")

    final_df = df[FINAL_COLUMNS].copy()
    for col in ["booked_on", "appt_date"]:
        final_df[col] = pd.to_datetime(final_df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    final_df.to_csv(FINAL_FILE, index=False)

    clinic_id = find_col(clinics, ["clinic_id"])
    name_col = find_col(clinics, ["clinic_name", "name"], required=False)
    city_col = find_col(clinics, ["city"], required=False)
    tier_col = find_col(clinics, ["tier"], required=False)
    lat_col = find_col(clinics, ["latitude", "lat"], required=False)
    lon_col = find_col(clinics, ["longitude", "lon", "lng"], required=False)

    clinic_cols = ["clinic_id", "clinic_name", "city", "tier", "latitude", "longitude", "temp_c"]
    sql = [
        "-- CareLoop Clinics – No-Show Guard: Part 1 generated SQL",
        "PRAGMA foreign_keys = ON;",
        "",
        "CREATE TABLE IF NOT EXISTS clinics (",
        " clinic_id TEXT PRIMARY KEY, clinic_name TEXT, city TEXT, tier TEXT,",
        " latitude REAL, longitude REAL, temp_c REAL",
        ");",
        "",
        "CREATE TABLE IF NOT EXISTS appointments (",
        " appointment_id TEXT PRIMARY KEY, patient_id TEXT, clinic_id TEXT,",
        " booked_on TEXT, appt_date TEXT, age INTEGER, gender TEXT,",
        " distance_km REAL, fee_inr REAL, booking_channel TEXT, visit_type TEXT,",
        " reminder_sent TEXT, prior_no_shows INTEGER, status TEXT,",
        " lead_days INTEGER, no_show INTEGER,",
        " FOREIGN KEY (clinic_id) REFERENCES clinics(clinic_id)",
        ");",
        "",
        "-- Clinic records",
    ]

    for _, row in clinics.iterrows():
        values = [
            row[clinic_id],
            row[name_col] if name_col else None,
            row[city_col] if city_col else None,
            row[tier_col] if tier_col else None,
            row[lat_col] if lat_col else None,
            row[lon_col] if lon_col else None,
            row["temp_c"] if "temp_c" in clinics.columns else None,
        ]
        sql.append(
            f"INSERT OR REPLACE INTO clinics ({', '.join(clinic_cols)}) "
            f"VALUES ({', '.join(sql_value(v) for v in values)});"
        )

    sql.append("\n-- Appointment records")
    for _, row in final_df.iterrows():
        values = ", ".join(sql_value(row[col]) for col in FINAL_COLUMNS)
        sql.append(
            f"INSERT OR REPLACE INTO appointments ({', '.join(FINAL_COLUMNS)}) "
            f"VALUES ({values});"
        )

    SQL_FILE.write_text("\n".join(sql) + "\n", encoding="utf-8")

    print("Final appointment rows:", len(final_df))
    print("Final columns:", len(final_df.columns))
    print("Duplicate appointment IDs:", int(final_df["appointment_id"].duplicated().sum()))
    print("Missing values:\n", final_df.isna().sum())
    print("Clinic records:", len(clinics))
    print("Saved:", FINAL_FILE)
    print("Saved:", SQL_FILE)
    return final_df


def main():
    print("CARELOOP CLINICS – NO-SHOW GUARD | PART 1")
    appointments = task_1_1()
    appointments = task_1_2(appointments)
    appointments = task_1_3(appointments)
    appointments = task_1_4(appointments)
    clinics = task_1_5()
    final_df = task_1_6(appointments, clinics)

    print("\n" + "=" * 60)
    print("PART 1 TASKS 1.1–1.6 COMPLETED")
    print("Final appointment rows:", len(final_df))
    print("No-show records:", int(final_df["no_show"].sum()))
    print("Output folder:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
