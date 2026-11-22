# GitHub Branch & Pull Request Guide (Team B)

Follow these steps to push `team_b_lightgbm/` to a dedicated GitHub branch **without touching `main`**.

## 1. Sync `main` and Create the Team B Branch

```bash
git checkout main
git pull origin main
git checkout -b team-b/lightgbm-final
git branch --show-current
# Expected: team-b/lightgbm-final
```

## 2. Copy `team_b_lightgbm/` into the Repository Root

Unzip `team_b_lightgbm_github_bundle.zip` and place the `team_b_lightgbm/` folder directly at the root of the repository.

## 3. Verify Before Committing

```bash
cd team_b_lightgbm
python scripts/evaluate.py
cd ..
git status
```

Confirm that only `team_b_lightgbm/` appears as untracked/modified.

## 4. Commit and Push to the Feature Branch

```bash
git add team_b_lightgbm/
git commit -m "Add Team B finalized Shared LightGBM forecasts, evaluation script, and results"
git push -u origin team-b/lightgbm-final
```

## 5. Open a Pull Request on GitHub

Open the repository on GitHub, click **Compare & pull request** (`base: main` <- `compare: team-b/lightgbm-final`), and paste:

**Title:**
```text
[Team B] Add finalized Shared LightGBM forecasts & matched_all_models evaluation
```

**Description:**
```markdown
## Summary
Adds the finalized Team B (Shared LightGBM) September 2026 test forecasts, unified evaluation script, and benchmark results under `team_b_lightgbm/` without modifying `main`.

## Final Model Selection (Fixed P90 >= 4.0, RMSE & Busy-Day Priority)
- **Primary Model:** `lightgbm_l2` (`fixed_train_feb_aug`, 30d) — RMSE `1.1332`, Busy RMSE `2.5382`, Busy MAE `2.0877`, 100% coverage (`39,564` rows).
- **Daily-Refit Benchmark:** `lightgbm_poisson` (`rolling_daily_refit`, 90d) — RMSE `1.1383` (`39,207` rows) / `1.1356` (`39,466` 90d rows), `99.75%` coverage.
- **Best MAE Reference:** `lightgbm_l1` (`rolling_daily_refit`, 90d) — MAE `0.7291`, Busy RMSE `2.8111`.
- **Supplementary & Baselines:** Includes `lightgbm_l2` (`rolling_daily_refit`) and `baseline_mean7` / `baseline_lag7` on the 36,483-row matched set.

## Verification
- Verified via `python team_b_lightgbm/scripts/evaluate.py` (0 duplicate keys, exact `y_true` match across all configurations).
- Compressed forecast file (`data/forecasts_september.csv.gz`) is ~3.4 MB (safe for standard Git push).
```
