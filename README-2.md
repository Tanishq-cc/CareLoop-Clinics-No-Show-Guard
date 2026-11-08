
# CareLoop Clinics – No-Show Guard

## 1. Project Summary

Part 1 – Data Cleaning:
[Add one important finding from data cleaning.]

Part 2 – SQL Analysis:
[Add one important finding from SQL analysis.]

Part 3 – Machine Learning:
[Add the selected model and its test ROC-AUC.]

Part 4 – Generative AI:
[Add the patient no-show reason distribution and API analysis result.]

## 2. Business Recommendation

CareLoop Clinics should introduce a targeted appointment reminder
and patient follow-up strategy.

Supporting findings:
- Total cleaned appointments: [990]
- Overall no-show rate: [22.02%]
- Highest no-show clinic rate: [Insert actual percentage]
- Highest-risk lead-time group: [Insert actual percentage]
- Most common patient-reported reason: [Insert actual category and count]

The clinics should prioritize patients with longer booking lead
times, improve reminder delivery, and provide support for
transportation, affordability, and scheduling difficulties.

## 3. Machine Learning Model

Selected model: [Insert selected model]

Test ROC-AUC: [Insert actual ROC-AUC]

Selected probability threshold: [Insert actual threshold]

Reason for selection:
The model was selected based on cross-validation performance
and its ability to identify patients at risk of missing
appointments. Recall and ROC-AUC were considered alongside
accuracy.

## 4. Limitations

1. Patient notes are illustrative and may not represent
   actual clinic-wide reasons for missed appointments.

2. A single random train-test split may not reflect
   performance on future appointments.

3. The dataset is observational, so associations between
   variables do not establish causation.

## 5. Technologies Used

- Python
- Pandas and NumPy
- SQLite
- Scikit-learn
- Google Colab
- Gemini API
- Pydantic
- GitHub

## 6. Project Files

- appointments_clean.csv
- clinics_enriched.csv
- notes_analysed_final.csv
- careloop_merged_analysis.csv
- root_cause_reminder_crosstab.csv
- test_risk_scores.csv
- load_data.sql
- 01_ingest_clean.ipynb
