# Report figures

Run from the repository root with the project requirements installed:

```bash
python src/plot_results.py
```

Each numbered figure has a PNG for slides, a vector PDF for the report and a CSV with the plotted values. Plotting uses the tracked daily panel, cleaning audits and published predictions. It does not retrain models. Model comparisons assert unique keys and matching actual targets.

| Figure | Purpose and interpretation |
|---|---|
| 01_input_sizes | Source workbook row counts before cleaning, reconstructed from ingestion audits. Monthly registry size varies. These are row counts, which can differ from distinct device counts. Session files may use different reporting conventions. |
| 02_session_flags | Flags for invalid energy/duration and review candidates across all cleaned session records. Categories can overlap. Counts alone do not imply every flagged session should be removed for the session-start target. |
| 03_label_coverage | Device-day coverage states by month after cleaning. Missing labels stay unknown. The usable state relies on the documented within-month logging assumption. |
| 04_training_patterns | February–June training target distribution and weekday means. Counts of 10 or more are pooled. Zero targets include inferred zeros. Means describe the observed cohort; weekday effects are not causal. |
| 05_model_comparison | RQ1, MAE and RMSE on the same 36,483 September device-days. Both axes start at zero. LightGBM has the smallest errors. Poisson improves RMSE over mean7 while its MAE is slightly worse. |
| 06_shared_local | RQ2, primary Poisson scores. Within each 30/90-day window, shared and local use matching rows. The populations differ between windows (39,513 and 39,550 September rows). This is a comparison of those Poisson specifications. |
| 07_busy_day_errors | RQ3, 3,281 matched September rows with actual sessions at least four. RMSE measures error size, bias measures signed error, and underprediction rate counts negative errors. Fewer underpredictions do not imply smaller errors. Conditioning on high realized counts emphasizes upward surprises. |
| 08_actual_groups | Mean prediction within groups defined by realized counts. Actual values at least eight are pooled. This descriptive figure illustrates shrinkage at high realized usage. Grouping by outcomes does not establish forecast calibration. |
| 09_device_example | One device with maximum matched-date coverage, nearest to median actual usage among those devices. Selection is descriptive and uses test outcomes. It is not selected for model performance and is not representative evidence by itself. Gaps indicate dates missing from the common evaluation set. |

## Report selection

For a short report, prioritize figures 03, 04, 05, 06 and 07. Use 01/02 in the data-quality discussion and 08/09 in an appendix or presentation.

## Scope and limitations

Original XLSX session files are excluded from Git and were unavailable when these figures were produced. Figure 01 describes the original inputs using their recorded audits. The package does not contain raw session energy/duration histograms, raw missingness heatmaps or original-row previews. Those require the exact workbooks identified in `config/input_manifest.json` or the corresponding session-level source table.

Figures inherit retrospective cohort selection, assumed zeros, source date uncertainty and model selection limitations described in the project README. The shared/local Poisson panel uses its paired eligible population, while cross-family comparisons use the common five-method population. The test month was inspected during development.
