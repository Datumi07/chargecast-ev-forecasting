# Data preparation and quality checks

Clean v1 supports next-day forecasting of recorded sessions per charging device. It covers September 2025 through September 2026. A device is identified by `com_id`, and the analysis uses the calendar date written in the source timestamps.

## Preparation rules

1. Read the 13 session exports and 13 device reports listed in [the input manifest](../config/input_manifest.json). The ingestion audit records row counts, exclusions and SHA-256 fingerprints.
2. Exclude session rows with a missing device ID or an unreadable start time. Store IDs as strings, trim whitespace, uppercase letters and remove a trailing `.0`. Preserve prefixes such as `APT` and `UC`, and log unmatched IDs for review.
3. Deduplicate on device, start time, site name, charging type, connector, currency, amount, energy and recorded duration. The exports lack a reliable session ID, so distinct sessions with identical fields may be merged. Missing connector IDs reduce confidence in these matches. The duplicate log records the source workbook and row for each removed copy.
4. Count eligible session records even if energy is zero, missing or invalid. For energy analysis, leave cleaned values blank when the original value is missing, negative or above 500 kWh. The last threshold is a review rule. Keep the original values for inspection.
5. Convert durations to hours while preserving elapsed days in Excel values. Mark missing or negative durations and durations above 365 days as invalid. Flag durations above 48 hours for review. Duration is excluded from the initial feature set.
6. Keep device IDs present in all 13 monthly reports. Of 1,756 matching registry IDs, 1,678 have at least one matching session. The other 78 remain in the cohort audit. Keep monthly reliability on its original 0–100 scale for auditing.
7. Build a continuous daily calendar for the 1,678 devices. In a device-month with at least one recorded session, infer zero for dates with no session rows. Logging gaps within the month can also produce apparent zeros. Months with no matching session rows have missing targets, which can also exclude genuine zero-usage months.
8. Mask labels for August 1–2, 2026 because of a network-wide gap. Also mask the first and last day of each month and the daylight saving transition dates of October 26, 2025 and March 29, 2026 as a precaution. The timezone remains unverified, so the mask cannot guarantee alignment with local Scottish calendar days.
9. Keep monthly reliability and service tickets out of the initial model features. Monthly reliability may be published after the forecast date, and an open ticket alone does not establish downtime.

## Clean v1 coverage

The panel contains 662,810 rows across 1,678 devices and 395 calendar days. There are 550,274 usable labels under the coverage rules, including 247,818 inferred zeros. The remaining rows contain missing targets. The final quality status counts each row once.

| Status | Device-days |
|---|---:|
| Assumed observed | 550,274 |
| Unknown device-month | 63,874 |
| Uncertain time boundary | 45,306 |
| Known network gap | 3,356 |

The recommended experiment starts in February 2026. Earlier months remain available for exploration and sensitivity analysis because detailed session counts differ from the monthly summaries. Selecting devices that remain in all reports creates a retrospective cohort and limits how broadly results can be generalized.

| Split | Period | Usable labels | Devices |
|---|---|---:|---:|
| Train | February–June 2026 | 209,496 | 1,623 |
| Validation | July–August 2026 | 81,912 | 1,506 |
| Test | September 2026 | 39,564 | 1,413 |

## Preparing inputs for forecasting

Use `sessions_target` as the outcome and apply `target_valid` when selecting labels. Create lags on the continuous calendar before dropping rows. Leave unavailable history missing. Apply the same calendar window and evaluation rows to the shared and device-specific models for each 30-day and 90-day experiment.

Keep the audit fields out of the predictor set. `observed_session_rows` reveals the same-day count and `month_session_rows` includes activity later in the reporting month. During evaluation, report the number of devices and days retained, the share of missing labels and the share of inferred zeros.

See the [processed data README](../data/processed/README.md) for file roles, the [raw data README](../data/raw/README.md) for source setup and the [root README](../README.md) for the rebuild command. The [audit summary](../reports/data_quality/audit_summary.json) records the validation checks.
