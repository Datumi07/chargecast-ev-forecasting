# Processed data
`daily_panel.csv.gz` is the main modelling table. Pandas reads gzip automatically. Preserve `com_id` as string and `sessions_target` as nullable integer. `registry_monthly.csv` is audit metadata, not contemporaneously available model input.

`sessions_clean.csv.gz` is bundled in the delivery ZIP for traceability but intentionally ignored by Git because it is larger and reproducible. The smaller daily panel and registry are explicitly allowed by `.gitignore`.

Do not use every column as X. Quality fields and same-day counts leak the target. See the root README and docs/data_cleaning.md.
