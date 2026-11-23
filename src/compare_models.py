"""Score the exported models on identical September device-day keys."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["com_id", "reported_date"]


def load_matched_forecasts():
    poisson = pd.read_csv(ROOT / "reports/poisson/forecasts.csv", dtype={"com_id": str})
    poisson = poisson.loc[(poisson.split == "test") & (poisson.history_days == 90) &
                          (poisson.training_scope == "device_specific")]
    ridge = pd.read_csv(ROOT / "reports/ridge/forecasts.csv.gz", dtype={"com_id": str})
    lightgbm = pd.read_csv(ROOT / "team_b_lightgbm/data/forecasts_september.csv.gz",
                          dtype={"com_id": str})
    candidates = {"poisson_local90": poisson, "ridge": ridge,
                  "lightgbm_fixed": lightgbm.loc[(lightgbm.model == "lightgbm_l2") &
                      (lightgbm.training_mode == "fixed_train_feb_aug") & (lightgbm.window_days == 30)],
                  "lag7": lightgbm.loc[lightgbm.model == "baseline_lag7"],
                  "mean7": lightgbm.loc[lightgbm.model == "baseline_mean7"]}
    combined = None
    coverage = []
    for name, frame in candidates.items():
        usable = frame[KEYS + ["y_true", "y_pred"]].dropna()
        assert not usable.duplicated(KEYS).any()
        coverage.append({"model": name, "available_rows": len(usable)})
        usable = usable.rename(columns={"y_pred": name})
        if combined is None:
            combined = usable
        else:
            combined = combined.merge(usable, on=KEYS, suffixes=("", "_check"), validate="one_to_one")
            assert combined.y_true.eq(combined.pop("y_true_check")).all()
    return combined, coverage, list(candidates)


def main():
    combined, coverage, candidates = load_matched_forecasts()
    results = []
    for name in candidates:
        error = combined[name] - combined.y_true
        busy = error[combined.y_true >= 4]
        results.append({"model": name, "rows": len(error), "mae": error.abs().mean(),
                        "rmse": np.sqrt(error.pow(2).mean()), "busy_rows": len(busy),
                        "busy_rmse": np.sqrt(busy.pow(2).mean()), "busy_bias": busy.mean(),
                        "busy_under_rate": (busy < 0).mean()})
    output = ROOT / "reports/comparison"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
