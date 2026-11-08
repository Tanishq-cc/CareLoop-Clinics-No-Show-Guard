-- ============================================================
-- CareLoop Clinics – No-Show Guard
-- PART 2: SQL TASKS 2.1–2.5
-- Database: SQLite
--
-- IMPORTANT:
-- 1. Run this script in SQLite (DB Browser for SQLite / sqlite3).
-- 2. First import clinics_enriched.csv into a table named clinics.
-- 3. Then import appointments_clean.csv into a table named appointments.
-- 4. The table columns must match the schemas documented below.
--
-- If you are using the Python file part2_sql_analysis.py, it creates
-- the database, tables, imports the CSV files, and runs these queries.
-- ============================================================

PRAGMA foreign_keys = ON;

-- TASK 2.1: Create the database tables.
-- Run these CREATE statements in a new/empty SQLite database.

CREATE TABLE IF NOT EXISTS clinics (
    clinic_id TEXT PRIMARY KEY,
    clinic_name TEXT,
    city TEXT,
    tier TEXT,
    latitude REAL,
    longitude REAL,
    temp_c REAL
);

CREATE TABLE IF NOT EXISTS appointments (
    appointment_id TEXT PRIMARY KEY,
    patient_id TEXT,
    clinic_id TEXT,
    booked_on TEXT,
    appt_date TEXT,
    age INTEGER,
    gender TEXT,
    distance_km REAL,
    fee_inr REAL,
    booking_channel TEXT,
    visit_type TEXT,
    reminder_sent TEXT,
    prior_no_shows INTEGER,
    status TEXT,
    lead_days INTEGER,
    no_show INTEGER,
    FOREIGN KEY (clinic_id) REFERENCES clinics(clinic_id)
);

-- Check loaded row counts after importing both CSV files.
SELECT 'clinics' AS table_name, COUNT(*) AS row_count FROM clinics
UNION ALL
SELECT 'appointments', COUNT(*) FROM appointments;

-- Check total no-shows.
SELECT
    COUNT(*) AS total_appointments,
    SUM(no_show) AS total_no_shows,
    ROUND(100.0 * SUM(no_show) / COUNT(*), 2) AS overall_no_show_pct
FROM appointments;


-- TASK 2.2: Clinic-wise no-show rate.
-- Include only clinics with at least 95 appointments.

SELECT
    c.clinic_name,
    c.city,
    COUNT(*) AS total_appointments,
    SUM(a.no_show) AS no_shows,
    ROUND(100.0 * SUM(a.no_show) / COUNT(*), 1) AS no_show_pct
FROM appointments AS a
JOIN clinics AS c
    ON a.clinic_id = c.clinic_id
GROUP BY c.clinic_id, c.clinic_name, c.city
HAVING COUNT(*) >= 95
ORDER BY no_show_pct DESC;


-- TASK 2.3: No-show rate by lead-time band.

SELECT
    CASE
        WHEN lead_days BETWEEN 1 AND 7 THEN '1-7 days'
        WHEN lead_days BETWEEN 8 AND 14 THEN '8-14 days'
        WHEN lead_days BETWEEN 15 AND 30 THEN '15-30 days'
        WHEN lead_days >= 31 THEN '31+ days'
        ELSE 'Invalid/0 days'
    END AS lead_time_band,
    COUNT(*) AS total_appointments,
    SUM(no_show) AS no_shows,
    ROUND(100.0 * SUM(no_show) / COUNT(*), 1) AS no_show_pct
FROM appointments
GROUP BY
    CASE
        WHEN lead_days BETWEEN 1 AND 7 THEN 1
        WHEN lead_days BETWEEN 8 AND 14 THEN 2
        WHEN lead_days BETWEEN 15 AND 30 THEN 3
        WHEN lead_days >= 31 THEN 4
        ELSE 5
    END
ORDER BY
    CASE
        WHEN lead_days BETWEEN 1 AND 7 THEN 1
        WHEN lead_days BETWEEN 8 AND 14 THEN 2
        WHEN lead_days BETWEEN 15 AND 30 THEN 3
        WHEN lead_days >= 31 THEN 4
        ELSE 5
    END;


-- TASK 2.4: Monthly no-show trend and month-over-month change.
-- LAG() retrieves the previous month's no-show percentage.

WITH monthly AS (
    SELECT
        strftime('%Y-%m', appt_date) AS appointment_month,
        COUNT(*) AS total_appointments,
        SUM(no_show) AS no_shows,
        ROUND(100.0 * SUM(no_show) / COUNT(*), 1) AS no_show_pct
    FROM appointments
    WHERE appt_date IS NOT NULL
    GROUP BY strftime('%Y-%m', appt_date)
),
trend AS (
    SELECT
        appointment_month,
        total_appointments,
        no_shows,
        no_show_pct,
        LAG(no_show_pct) OVER (ORDER BY appointment_month) AS previous_month_pct
    FROM monthly
)
SELECT
    appointment_month,
    total_appointments,
    no_shows,
    no_show_pct,
    CASE
        WHEN previous_month_pct IS NULL THEN NULL
        ELSE ROUND(no_show_pct - previous_month_pct, 1)
    END AS change_percentage_points
FROM trend
ORDER BY appointment_month;


-- TASK 2.5: Rank clinics within each city and compare with overall rate.

WITH clinic_rates AS (
    SELECT
        c.clinic_id,
        c.clinic_name,
        c.city,
        COUNT(*) AS total_appointments,
        SUM(a.no_show) AS no_shows,
        100.0 * SUM(a.no_show) / COUNT(*) AS no_show_pct
    FROM appointments AS a
    JOIN clinics AS c
        ON a.clinic_id = c.clinic_id
    GROUP BY c.clinic_id, c.clinic_name, c.city
),
overall AS (
    SELECT
        100.0 * SUM(no_show) / COUNT(*) AS overall_no_show_pct
    FROM appointments
)
SELECT
    clinic_name,
    city,
    total_appointments,
    no_shows,
    ROUND(no_show_pct, 1) AS no_show_pct,
    RANK() OVER (
        PARTITION BY city
        ORDER BY no_show_pct DESC
    ) AS rank_within_city,
    ROUND(overall.overall_no_show_pct, 2) AS overall_no_show_pct,
    ROUND(no_show_pct - overall.overall_no_show_pct, 1)
        AS difference_from_overall_pp
FROM clinic_rates
CROSS JOIN overall
ORDER BY city, rank_within_city, clinic_name;
