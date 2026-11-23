"""Generate report figures and their numerical source tables from tracked inputs."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compare_models import load_matched_forecasts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/figures"
LABELS = {"poisson_local90": "Poisson local 90d", "ridge": "Ridge",
          "lightgbm_fixed": "LightGBM fixed", "lag7": "Same weekday last week",
          "mean7": "Previous 7-day mean"}
COLORS = ["#5268a0", "#c17d26", "#208572", "#9c6688", "#78858d"]


def save(fig, name, table):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    table.to_csv(OUT / f"{name}.csv", index=False)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    panel = pd.read_csv(ROOT / "data/processed/daily_panel.csv.gz", dtype={"com_id": str},
                        parse_dates=["reported_date"])
    audit = pd.read_csv(ROOT / "reports/data_quality/ingestion_audit.csv")
    monthly = audit.groupby(["month", "kind"]).raw_rows.sum().unstack().sort_index()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, kind, title in zip(axes, ["sessions", "uptime"],
                              ["Input session rows", "Input monthly registry rows"]):
        monthly[kind].plot.bar(ax=ax, color="#208572")
        ax.set_title(title)
        ax.set_ylabel("Rows")
        ax.set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("Before cleaning — source workbook sizes from ingestion audits")
    save(fig, "01_input_sizes", monthly.reset_index())

    quality = json.loads((ROOT / "reports/data_quality/session_quality_summary.json").read_text())
    counts = pd.Series(quality).sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(counts.index.str.replace("_", " "), counts, color="#c17d26")
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.set_xlim(0, counts.max() * 1.2)
    ax.set_xlabel("Flagged records across all cleaned session records")
    ax.set_title("Session quality flags — categories can overlap")
    save(fig, "02_session_flags", counts.rename("rows").rename_axis("flag").reset_index())

    coverage = panel.assign(month=panel.reported_date.dt.strftime("%Y-%m")).groupby(
        ["month", "coverage_status"]).size().unstack(fill_value=0)
    coverage = coverage.div(coverage.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    coverage.plot.bar(stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("Device-days (%)")
    ax.set_title("After cleaning — usable and masked labels by month")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Coverage status", bbox_to_anchor=(1.02, 1))
    save(fig, "03_label_coverage", coverage.reset_index())

    train = panel.loc[panel.recommended_period & panel.target_valid & panel.split.eq("train")]
    dist = train.sessions_target.clip(upper=10).value_counts().sort_index()
    weekdays = train.groupby(train.reported_date.dt.weekday).sessions_target.agg(["mean", "count"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(dist.index, dist / dist.sum() * 100, color="#5268a0")
    axes[0].set_xticks(range(11), [str(i) for i in range(10)] + ["10+"])
    axes[0].set_xlabel("Recorded sessions per device-day")
    axes[0].set_ylabel("Usable training rows (%)")
    axes[0].set_title("Zero counts and the upper tail")
    axes[1].bar(range(7), weekdays["mean"], color="#208572")
    axes[1].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1].set_ylabel("Mean sessions per usable device-day")
    axes[1].set_title("Weekday pattern")
    save(fig, "04_training_patterns", pd.concat([
        dist.rename("rows").rename_axis("sessions_capped_at_10").reset_index(),
        weekdays.rename_axis("weekday").reset_index()], axis=1))

    matched, _, names = load_matched_forecasts()
    rows = []
    for name in names:
        error = matched[name] - matched.y_true
        busy = error.loc[matched.y_true >= 4]
        rows.append(dict(model=name, mae=error.abs().mean(), rmse=np.sqrt((error**2).mean()),
                         busy_rmse=np.sqrt((busy**2).mean()), busy_bias=busy.mean(),
                         underprediction=100 * (busy < 0).mean()))
    scores = pd.DataFrame(rows).set_index("model").loc[names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["mae", "rmse"]):
        bars = ax.barh([LABELS[n] for n in names], scores[metric], color=COLORS)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_xlim(0, scores[metric].max() * 1.18)
        ax.set_title(metric.upper() + " — lower is better")
        ax.set_xlabel("Sessions per device-day")
        ax.invert_yaxis()
    fig.suptitle(f"RQ1 — identical {len(matched):,} September device-days")
    save(fig, "05_model_comparison", scores.reset_index())

    rq2 = pd.read_csv(ROOT / "reports/poisson/metrics.csv").query("split == 'test'")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["mae", "rmse"]):
        pivot = rq2.pivot(index="history_days", columns="training_scope", values=metric)
        pivot.rename(columns={"device_specific": "Local", "shared": "Shared"}).plot.bar(
            ax=ax, color=["#208572", "#5268a0"], rot=0)
        ax.set_xlabel("Training history (calendar days)")
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper() + " — lower is better")
    fig.suptitle("RQ2 — paired Poisson comparison; matched keys within each window")
    save(fig, "06_shared_local", rq2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, title in zip(axes, ["busy_rmse", "busy_bias", "underprediction"],
            ["Busy RMSE — lower is better", "Busy bias — closer to zero", "Busy underprediction (%)"]):
        ax.barh([LABELS[n] for n in names], scores[metric], color=COLORS)
        ax.set_title(title)
        ax.axvline(0, color="#333333", linewidth=.6)
        ax.invert_yaxis()
    fig.suptitle("RQ3 — actual sessions ≥ 4; selecting high realized counts emphasizes upward surprises")
    save(fig, "07_busy_day_errors", scores.reset_index())

    calibration = matched.assign(actual_group=matched.y_true.clip(upper=8)).groupby("actual_group")[
        ["y_true"] + names].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = calibration[("y_true", "mean")]
    ax.plot(x, x, "--", color="#555555", label="Perfect group mean")
    for name, color in zip(names, COLORS):
        ax.plot(x, calibration[(name, "mean")], marker="o", color=color, label=LABELS[name])
    ax.set_xlabel("Mean actual sessions in each group (counts ≥ 8 pooled)")
    ax.set_ylabel("Mean prediction")
    ax.set_title("Prediction versus realized count — descriptive groups, September")
    ax.legend()
    table = calibration.copy()
    table.columns = ["_".join(c) for c in table.columns]
    save(fig, "08_actual_groups", table.reset_index())

    # Choose by complete matched-date coverage, then median actual volume.
    by_device = matched.groupby("com_id").y_true.agg(["size", "mean"])
    eligible = by_device.loc[by_device["size"] == by_device["size"].max()].sort_index()
    selected = (eligible["mean"] - eligible["mean"].median()).abs().idxmin()
    example = matched.loc[matched.com_id == selected].sort_values("reported_date").copy()
    example.reported_date = pd.to_datetime(example.reported_date)
    example = example.set_index("reported_date").reindex(
        pd.date_range("2026-09-01", "2026-09-30", name="reported_date")).reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(example.reported_date, example.y_true, color="#222222", marker="o", label="Actual")
    for name, color in zip(names[:3], COLORS[:3]):
        ax.plot(example.reported_date, example[name], color=color, label=LABELS[name])
    ax.set_title(f"Example device {selected} — maximum matched coverage, median usage")
    ax.set_ylabel("Recorded sessions")
    ax.legend(ncol=4)
    fig.autofmt_xdate()
    save(fig, "09_device_example", example)


if __name__ == "__main__":
    main()
