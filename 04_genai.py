"""
CareLoop Clinics – No-Show Guard
PART 4: GENERATIVE AI AND PATIENT NOTE ANALYSIS (Tasks 4.1–4.4)

Input files expected beside this script:
    followup_notes.csv
    appointments_clean.csv

Optional existing Task 4.1 output:
    notes_analysed_final.csv

Install:
    pip install pandas pydantic google-genai scikit-learn

Set your Gemini API key before running:
    macOS/Linux:
        export GEMINI_API_KEY="YOUR_KEY"
    Windows PowerShell:
        $env:GEMINI_API_KEY="YOUR_KEY"

Run:
    python part4_generative_ai.py

SECURITY:
- Never commit your API key or a .env file containing it to GitHub.
- This script reads GEMINI_API_KEY from the environment.
- API calls may incur charges or be subject to free-tier quotas.
- If quota is exhausted, the script records the failure honestly and continues.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise SystemExit(
        "Missing Gemini SDK. Install it with: pip install google-genai"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs_part4"
OUTPUT_DIR.mkdir(exist_ok=True)

NOTES_FILE = BASE_DIR / "followup_notes.csv"
APPOINTMENTS_FILE = BASE_DIR / "appointments_clean.csv"
TASK_4_1_FILE = OUTPUT_DIR / "notes_analysed_final.csv"
TASK_4_2_FILE = OUTPUT_DIR / "task_4_2_validation.csv"
TASK_4_3_FILE = OUTPUT_DIR / "task_4_3_prompt_comparison.csv"
MERGED_FILE = OUTPUT_DIR / "careloop_merged_analysis.csv"
CROSSTAB_FILE = OUTPUT_DIR / "root_cause_reminder_crosstab.csv"
README_FILE = OUTPUT_DIR / "README_part4_findings.md"

MODEL_NAME = "gemini-3.6-flash"
CATEGORIES = ["Forgot", "Distance", "Cost", "Illness", "Work", "Other"]


class PatientAnalysis(BaseModel):
    reason_category: Literal[
        "Forgot", "Distance", "Cost", "Illness", "Work", "Other"
    ]
    reason_summary: str = Field(min_length=5)
    confidence: float = Field(ge=0, le=1)


class FollowupInsight(BaseModel):
    root_cause: Literal[
        "Forgot", "Distance", "Cost", "Illness", "Work", "Other"
    ]
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=5)
    recommended_action: str = Field(min_length=5)


def load_inputs():
    if not NOTES_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {NOTES_FILE.name}")
    if not APPOINTMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {APPOINTMENTS_FILE.name}")

    notes = pd.read_csv(NOTES_FILE)
    appointments = pd.read_csv(APPOINTMENTS_FILE)

    required_notes = {"note_id", "appointment_id", "note_text"}
    missing = required_notes - set(notes.columns)
    if missing:
        raise ValueError(f"followup_notes.csv is missing columns: {sorted(missing)}")

    if "appointment_id" not in appointments.columns:
        raise ValueError("appointments_clean.csv needs an appointment_id column")

    return notes, appointments


def make_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it locally; do not paste the key into this script."
        )
    return genai.Client(api_key=api_key)


def call_gemini(client, prompt, schema):
    """Call Gemini with a structured Pydantic response schema."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    if response.parsed is not None:
        parsed = response.parsed
        if isinstance(parsed, schema):
            return parsed
        return schema.model_validate(parsed)

    return schema.model_validate_json(response.text)


def task_4_1(notes, client):
    print("\nTASK 4.1 – GEMINI API NOTE CLASSIFICATION")
    results = []
    api_calls = 0

    for _, row in notes.iterrows():
        note = row["note_text"]
        record = {
            "note_id": row["note_id"],
            "appointment_id": row["appointment_id"],
            "note_text": note,
            "reason_category": None,
            "reason_summary": None,
            "confidence": None,
            "analysis_status": None,
            "error_message": None,
        }

        if pd.isna(note) or not str(note).strip():
            record["reason_category"] = "Other"
            record["reason_summary"] = "No patient note was provided."
            record["confidence"] = 0.0
            record["analysis_status"] = "Skipped"
            results.append(record)
            continue

        prompt = f"""
You are analysing a clinic appointment no-show note.
Classify the reason into exactly one category:
Forgot, Distance, Cost, Illness, Work, Other.

Return a short evidence-based summary and confidence from 0 to 1.
Do not invent facts. If the reason is unclear, choose Other.

Patient note:
{note}
"""
        try:
            api_calls += 1
            answer = call_gemini(client, prompt, PatientAnalysis)
            record.update({
                "reason_category": answer.reason_category,
                "reason_summary": answer.reason_summary,
                "confidence": answer.confidence,
                "analysis_status": "Success",
            })
            print(f"{row['note_id']}: Success")
        except Exception as exc:
            record["analysis_status"] = "Failed"
            record["error_message"] = str(exc)[:500]
            print(f"{row['note_id']}: API/validation failure: {exc}")

        results.append(record)
        time.sleep(1)

    result_df = pd.DataFrame(results)
    result_df.to_csv(TASK_4_1_FILE, index=False)
    print("Saved:", TASK_4_1_FILE)
    print("Gemini requests attempted in Task 4.1:", api_calls)
    return result_df, api_calls


def task_4_2(notes, task41):
    print("\nTASK 4.2 – PYDANTIC VALIDATION OF EXISTING RESULTS")
    # Validate only successful Task 4.1 outputs. Do not relabel quota failures
    # as valid AI responses and do not silently replace missing values.
    results = []

    for _, note_row in notes.head(10).iterrows():
        note_id = note_row["note_id"]
        matched = task41[
            task41["note_id"].astype(str) == str(note_id)
        ]

        if matched.empty:
            results.append({
                "note_id": note_id,
                "validation_status": "missing_task_4_1_result",
                "root_cause": None,
                "confidence": None,
                "summary": None,
            })
            continue

        old = matched.iloc[0]
        if old.get("analysis_status") != "Success":
            results.append({
                "note_id": note_id,
                "validation_status": "api_or_analysis_failed",
                "root_cause": None,
                "confidence": None,
                "summary": None,
            })
            continue

        try:
            validated = PatientAnalysis(
                reason_category=old["reason_category"],
                reason_summary=old["reason_summary"],
                confidence=float(old["confidence"]),
            )
            results.append({
                "note_id": note_id,
                "validation_status": "valid",
                "root_cause": validated.reason_category,
                "confidence": validated.confidence,
                "summary": validated.reason_summary,
            })
        except (ValidationError, TypeError, ValueError) as exc:
            results.append({
                "note_id": note_id,
                "validation_status": "validation_error",
                "root_cause": None,
                "confidence": None,
                "summary": str(exc)[:300],
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(TASK_4_2_FILE, index=False)
    print(result_df.to_string(index=False))
    print("Saved:", TASK_4_2_FILE)
    return result_df


def task_4_3(notes, client):
    print("\nTASK 4.3 – PYDANTIC FAILURE TESTS AND PROMPT COMPARISON")

    # Deliberately invalid examples: both must raise ValidationError.
    invalid_examples = [
        {
            "root_cause": "Forgot",
            "confidence": 1.7,
            "summary": "Patient forgot the appointment.",
            "recommended_action": "Send a reminder."
        },
        {
            "root_cause": "Distance",
            "confidence": 0.8,
            "summary": "Patient lives far away."
            # recommended_action intentionally missing
        },
    ]

    print("\nPydantic negative tests:")
    for number, bad_data in enumerate(invalid_examples, start=1):
        try:
            FollowupInsight(**bad_data)
            print(f"Test {number}: Unexpectedly passed")
        except ValidationError:
            print(f"Test {number}: ValidationError caught successfully")

    selected = notes[
        notes["note_id"].astype(str).isin(["N007", "N009"])
    ]

    if selected.empty:
        print("N007/N009 not found; prompt comparison was not run.")
        return pd.DataFrame(), 0

    prompt_v1 = """
Classify this patient appointment no-show note into:
Forgot, Distance, Cost, Illness, Work, Other.
Give a short explanation and confidence between 0 and 1.
"""

    prompt_v2 = """
You are a careful clinic no-show analysis assistant.
Classify the note into exactly one category:
Forgot, Distance, Cost, Illness, Work, Other.
Understand English and Hinglish, interpret the full context,
do not infer facts not stated, and choose Other if the reason
cannot be established. Return a concise evidence-based summary
and a confidence score between 0 and 1.
"""

    comparison = []
    api_calls = 0

    for _, row in selected.iterrows():
        for version, prompt in [("V1", prompt_v1), ("V2", prompt_v2)]:
            item = {
                "note_id": row["note_id"],
                "note_text": row["note_text"],
                "prompt_version": version,
                "reason_category": None,
                "reason_summary": None,
                "confidence": None,
                "result_status": None,
                "error_message": None,
            }

            full_prompt = prompt + "\nPatient note:\n" + str(row["note_text"])
            try:
                api_calls += 1
                answer = call_gemini(client, full_prompt, PatientAnalysis)
                item.update({
                    "reason_category": answer.reason_category,
                    "reason_summary": answer.reason_summary,
                    "confidence": answer.confidence,
                    "result_status": "Success",
                })
            except Exception as exc:
                item["result_status"] = "Failed"
                item["error_message"] = str(exc)[:500]
                print(f"{row['note_id']} {version}: {exc}")

            comparison.append(item)
            time.sleep(1)

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(TASK_4_3_FILE, index=False)
    print(comparison_df.to_string(index=False))
    print("Saved:", TASK_4_3_FILE)
    print("Task 4.3 Gemini requests attempted:", api_calls)
    return comparison_df, api_calls


def task_4_4(appointments, task41):
    print("\nTASK 4.4 – MERGE, CROSS-TABULATION AND RECOMMENDATIONS")

    notes_for_merge = task41.copy()
    notes_for_merge = notes_for_merge[
        ["appointment_id", "reason_category", "reason_summary", "confidence", "analysis_status"]
    ]

    merged = appointments.merge(
        notes_for_merge,
        on="appointment_id",
        how="left",
    )
    merged.to_csv(MERGED_FILE, index=False)

    if "reminder_sent" in merged.columns:
        cross_tab = pd.crosstab(
            merged["reason_category"],
            merged["reminder_sent"],
            margins=True,
        )
        cross_tab.to_csv(CROSSTAB_FILE)
        print("\nRoot cause vs reminder sent:")
        print(cross_tab.to_string())
    else:
        cross_tab = pd.DataFrame()
        print("reminder_sent column missing; cross-tabulation not created.")

    if "no_show" in merged.columns and "reason_category" in merged.columns:
        analysis = merged.dropna(subset=["reason_category"]).groupby(
            "reason_category"
        ).agg(
            appointments=("appointment_id", "count"),
            no_shows=("no_show", "sum"),
        )
        analysis["no_show_rate_pct"] = (
            100 * analysis["no_shows"] / analysis["appointments"]
        ).round(2)
        print("\nNo-show rate by AI-classified reason:")
        print(analysis.to_string())
    else:
        analysis = pd.DataFrame()

    # Write a results-based report; do not invent counts or findings.
    lines = [
        "# CareLoop Clinics – Part 4 Findings",
        "",
        "## Task 4.1 – Gemini API classification",
        f"- Patient notes in source file: {len(task41)}",
        f"- Successful classifications: {(task41['analysis_status'] == 'Success').sum()}",
        f"- Failed API/analysis records: {(task41['analysis_status'] == 'Failed').sum()}",
        f"- Skipped blank notes: {(task41['analysis_status'] == 'Skipped').sum()}",
        "",
        "## Task 4.2 – Pydantic validation",
        "- Successful Task 4.1 results for the first 10 notes were checked against the Pydantic schema.",
        "- Missing or failed API responses remain explicitly identified; they are not presented as valid AI results.",
        "",
        "## Task 4.3 – Prompt comparison",
        "- The V1/V2 comparison output is saved separately when Gemini requests succeed.",
        "- Pydantic negative tests check an out-of-range confidence and a missing required field.",
        "",
        "## Task 4.4 – Merge and analysis",
        f"- Appointment rows in merged dataset: {len(merged)}",
        f"- Rows with a successful AI classification: {int(merged['reason_category'].notna().sum())}",
        "",
        "## Limitations",
        "- AI classification depends on the wording and completeness of patient notes.",
        "- API quota limits or service errors can leave some notes unclassified.",
        "- Observational associations do not prove that a reported reason caused a missed appointment.",
        "",
        "## Generated files",
        "- notes_analysed_final.csv",
        "- task_4_2_validation.csv",
        "- task_4_3_prompt_comparison.csv",
        "- careloop_merged_analysis.csv",
        "- root_cause_reminder_crosstab.csv",
    ]
    README_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSaved:", MERGED_FILE)
    if not cross_tab.empty:
        print("Saved:", CROSSTAB_FILE)
    print("Saved:", README_FILE)
    return merged


def main():
    print("CARELOOP CLINICS – NO-SHOW GUARD")
    print("PART 4: GENERATIVE AI")

    notes, appointments = load_inputs()
    client = make_client()

    task41, calls41 = task_4_1(notes, client)
    task42 = task_4_2(notes, task41)
    comparison, calls43 = task_4_3(notes, client)
    merged = task_4_4(appointments, task41)

    print("\n" + "=" * 68)
    print("PART 4 SUMMARY")
    print("=" * 68)
    print("Task 4.1 API requests attempted:", calls41)
    print("Task 4.3 API requests attempted:", calls43)
    print("Total API requests attempted in this run:", calls41 + calls43)
    print("Task 4.2 validation rows:", len(task42))
    print("Task 4.3 comparison rows:", len(comparison))
    print("Merged appointment rows:", len(merged))
    print("Outputs folder:", OUTPUT_DIR)
    print("Note: attempted requests include requests that may have failed/quota-exhausted.")


if __name__ == "__main__":
    main()
