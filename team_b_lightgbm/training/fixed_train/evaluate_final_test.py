"""Chặng 5: Đánh giá chốt hạ 1 lần duy nhất trên tập Test (tháng 9/2026) & vẽ 3 biểu đồ báo cáo.

Quy ước chống rò rỉ dữ liệu:
- Siêu tham số và số cây dừng (`best_iteration`) được khóa cố định từ kết quả Chặng 4
  (`reports/modeling/best_lgbm_config.json`) — tuyệt đối không tinh chỉnh lại trên Test.
- Ngưỡng ngày cao điểm (top 10% = 4.0 phiên) vẫn tính duy nhất trên tập Train.
- Báo cáo song song cả 2 chế độ huấn luyện cho mô hình LightGBM chiến thắng:
  1. `lgbm_tuned_train_only`: Train trên tháng 2–6 (so sòng phẳng nếu nhóm khác chỉ train 2–6).
  2. `lgbm_tuned_trainval`: Train lại trên tháng 2–8 với cùng `best_iteration` đã chốt
     (thực hành chuẩn trong dự báo chuỗi thời gian để mô hình có dữ liệu hè tháng 7–8 sát tháng 9).

Đầu ra:
- reports/modeling/final_test_metrics.csv
- reports/modeling/final_test_feature_importance.csv
- reports/modeling/final_test_usage_buckets.csv
- reports/modeling/figures/01_test_model_comparison.png
- reports/modeling/figures/02_top10_feature_importance.png
- reports/modeling/figures/03_high_usage_error_analysis.png

Cách chạy từ thư mục gốc của repo:
    python team_b_lightgbm/training/fixed_train/evaluate_final_test.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import (
    compute_high_usage_threshold,
    evaluate_predictions,
    filter_split,
    load_feature_dataset,
)
from train_lgbm_shared import prepare_features


def fit_lgbm_fixed_trees(
    df_fit: pd.DataFrame,
    df_test: pd.DataFrame,
    predictors: list[str],
    params: dict[str, Any],
    n_trees: int,
) -> tuple[np.ndarray, lgb.LGBMRegressor, list[str]]:
    """Huấn luyện LightGBM với số cây cố định (không nhìn vào tập Test) và dự báo Test."""
    X_fit, X_test, cat_cols = prepare_features(
        df_fit, df_test, predictors, use_com_id=True
    )
    y_fit = df_fit["sessions_target"].to_numpy(dtype=float)

    clean_params = {
        k: v for k, v in params.items() if k != "n_estimators"
    }
    model = lgb.LGBMRegressor(
        n_estimators=int(n_trees),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **clean_params,
    )
    model.fit(X_fit, y_fit, categorical_feature=cat_cols)
    y_pred = np.clip(model.predict(X_test), 0.0, None)
    return y_pred, model, list(X_fit.columns)


def compute_usage_bucket_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    high_usage_threshold: float,
) -> pd.DataFrame:
    """Phân tích lỗi và tỷ lệ đoán thiếu theo từng nhóm mức sử dụng thực tế (trả lời RQ3)."""
    buckets = [
        ("0 sessions", y_true == 0),
        ("1 session", y_true == 1),
        ("2-3 sessions", (y_true >= 2) & (y_true < high_usage_threshold)),
        (f"{int(high_usage_threshold)}+ (High usage)", y_true >= high_usage_threshold),
    ]
    rows = []
    for label, mask in buckets:
        yt = y_true[mask]
        yp = y_pred[mask]
        err = yp - yt
        rows.append(
            {
                "usage_bucket": label,
                "n_rows": int(mask.sum()),
                "mean_actual": round(float(np.mean(yt)), 4),
                "mean_predicted": round(float(np.mean(yp)), 4),
                "mae": round(float(np.mean(np.abs(err))), 4),
                "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
                "bias": round(float(np.mean(err)), 4),
                "underpred_rate_pct": round(float(np.mean(err < 0) * 100.0), 2),
                "mean_underpred": round(float(np.mean(np.maximum(0.0, -err))), 4),
            }
        )
    return pd.DataFrame(rows)


def plot_model_comparison(test_df: pd.DataFrame, out_path: Path) -> None:
    """Biểu đồ 1: So sánh MAE & RMSE trên tập Test (cửa sổ 90d, all_eligible)."""
    sub = test_df[
        (test_df["window_days"] == 90) & (test_df["subset"] == "all_eligible")
    ].copy()

    label_map = {
        "baseline_same_weekday_last_week": "Baseline:\nSame Weekday",
        "baseline_historical_average_90d": "Baseline:\nHist. Avg 90d",
        "lgbm_shared_default": "LightGBM Shared\n(Default)",
        "lgbm_tuned_train_only": "LightGBM Tuned\n(Train 2-6)",
        "lgbm_tuned_trainval": "LightGBM Tuned\n(Train+Val 2-8)",
    }
    sub["label"] = sub["model"].map(lambda m: label_map.get(m, m))

    x = np.arange(len(sub))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    b1 = ax.bar(x - width / 2, sub["mae"], width, label="MAE", color="#2563EB")
    b2 = ax.bar(x + width / 2, sub["rmse"], width, label="RMSE", color="#0F172A")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_ylabel("Error (sessions / device-day, lower is better)", fontsize=10)
    ax.set_title("Test Set Performance Comparison (Sep 2026 | 90-day Window | All Eligible)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["label"], fontsize=9.5)
    ax.set_ylim(0, max(sub["rmse"].max() * 1.18, 1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(feat_imp: pd.DataFrame, out_path: Path) -> None:
    """Biểu đồ 2: Top 10 đặc trưng quan trọng nhất (% Gain)."""
    top10 = feat_imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bars = ax.barh(top10["feature"], top10["gain_pct"], color="#0D9488")

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.1f}%",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Share of Total Gain (%)", fontsize=10)
    ax.set_title("Top 10 Most Important Features — Final Tuned LightGBM (90d)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, top10["gain_pct"].max() * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_usage_buckets(bucket_df: pd.DataFrame, out_path: Path) -> None:
    """Biểu đồ 3: Phân tích lỗi và dự báo thiếu theo mức độ sử dụng (RQ3)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    x = np.arange(len(bucket_df))
    w = 0.36

    # Trái: Trung bình Thực tế vs Dự báo
    ax1.bar(x - w / 2, bucket_df["mean_actual"], w, label="Actual Mean", color="#1E293B")
    ax1.bar(x + w / 2, bucket_df["mean_predicted"], w, label="Predicted Mean", color="#3B82F6")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bucket_df["usage_bucket"], fontsize=9)
    ax1.set_ylabel("Mean Sessions / Device-Day", fontsize=10)
    ax1.set_title("Actual vs. Predicted by Usage Bucket", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False)

    # Phải: MAE & Tỷ lệ đoán thiếu theo nhóm
    bars = ax2.bar(x, bucket_df["mae"], width=0.5, color=["#94A3B8", "#64748B", "#F59E0B", "#DC2626"])
    for bar, rate in zip(bars, bucket_df["underpred_rate_pct"]):
        h = bar.get_height()
        ax2.annotate(
            f"MAE={h:.2f}\n(Under: {rate:.0f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(bucket_df["usage_bucket"], fontsize=9)
    ax2.set_ylabel("MAE (sessions / device-day)", fontsize=10)
    ax2.set_title("Error & Underprediction Rate by Usage Bucket", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, bucket_df["mae"].max() * 1.28)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("RQ3 Error Analysis on Test Set (Sep 2026) — Final Tuned LightGBM (90d)", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg_path = Path("reports/modeling/best_lgbm_config.json")
    if not cfg_path.exists():
        cfg_path = Path(__file__).with_name("best_lgbm_config.json")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {cfg_path}. Hãy chạy `python team_b_lightgbm/training/fixed_train/tune_lgbm.py` trước."
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        best_cfg = json.load(f)

    best_params = best_cfg["params"]
    if "best_iteration_90d" in best_cfg:
        best_iter_90 = int(best_cfg["best_iteration_90d"])
        best_iter_30 = int(best_cfg["best_iteration_30d"])
    else:
        best_iter_90 = int(best_cfg["validation_90d_all_eligible"]["best_iteration"])
        best_iter_30 = int(best_cfg["validation_30d_best_iteration"])

    # Cấu hình mặc định Chặng 3A (để so trực tiếp trên Test)
    default_params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
    }
    default_iters = {30: 100, 90: 109}
    tuned_iters = {30: best_iter_30, 90: best_iter_90}

    all_rows: list[dict[str, Any]] = []
    final_90_model: lgb.LGBMRegressor | None = None
    final_90_cols: list[str] = []
    final_90_y_true: np.ndarray | None = None
    final_90_y_pred: np.ndarray | None = None
    final_90_threshold = 4.0

    for window in (30, 90):
        df, predictors = load_feature_dataset(window)
        df_train = filter_split(df, "train", eligible_only=True)
        df_val = filter_split(df, "validation", eligible_only=True)
        df_test = filter_split(df, "test", eligible_only=True)
        df_trainval = pd.concat([df_train, df_val], ignore_index=True)

        threshold = compute_high_usage_threshold(df_train, quantile=0.90)

        # 1. Baseline: Same weekday last week
        all_rows.extend(
            evaluate_predictions(
                df_test, df_test["lag_7"], threshold,
                "baseline_same_weekday_last_week", window, "test"
            )
        )
        # 2. Baseline: Historical average
        all_rows.extend(
            evaluate_predictions(
                df_test, df_test[f"rolling_mean_{window}"], threshold,
                f"baseline_historical_average_{window}d", window, "test"
            )
        )
        # 3. LightGBM Shared Default (Chặng 3A, train 2-6)
        pred_def, _, _ = fit_lgbm_fixed_trees(
            df_train, df_test, predictors, default_params, default_iters[window]
        )
        all_rows.extend(
            evaluate_predictions(
                df_test, pred_def, threshold, "lgbm_shared_default", window, "test"
            )
        )
        # 4. LightGBM Shared Tuned (Train 2-6 only)
        pred_tuned_tr, _, _ = fit_lgbm_fixed_trees(
            df_train, df_test, predictors, best_params, tuned_iters[window]
        )
        all_rows.extend(
            evaluate_predictions(
                df_test, pred_tuned_tr, threshold, "lgbm_tuned_train_only", window, "test"
            )
        )
        # 5. LightGBM Shared Tuned (Refit Train+Val 2-8 với cùng best_iteration)
        pred_tuned_tv, model_tv, cols_tv = fit_lgbm_fixed_trees(
            df_trainval, df_test, predictors, best_params, tuned_iters[window]
        )
        all_rows.extend(
            evaluate_predictions(
                df_test, pred_tuned_tv, threshold, "lgbm_tuned_trainval", window, "test"
            )
        )

        if window == 90:
            final_90_model = model_tv
            final_90_cols = cols_tv
            final_90_y_true = df_test["sessions_target"].to_numpy(dtype=float)
            final_90_y_pred = pred_tuned_tv
            final_90_threshold = threshold

    out_dir = Path("reports/modeling")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.DataFrame(all_rows)
    test_csv = out_dir / "final_test_metrics.csv"
    test_df.to_csv(test_csv, index=False)

    # Feature importance của mô hình cuối cùng (90d tuned train+val)
    assert final_90_model is not None and final_90_y_true is not None and final_90_y_pred is not None
    gains = final_90_model.booster_.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame(
        {
            "feature": final_90_cols,
            "gain": gains,
            "gain_pct": 100.0 * gains / max(gains.sum(), 1e-12),
        }
    ).sort_values("gain", ascending=False)
    feat_csv = out_dir / "final_test_feature_importance.csv"
    feat_imp.to_csv(feat_csv, index=False)

    # Phân tích lỗi theo nhóm mức sử dụng (RQ3)
    bucket_df = compute_usage_bucket_breakdown(
        final_90_y_true, final_90_y_pred, final_90_threshold
    )
    bucket_csv = out_dir / "final_test_usage_buckets.csv"
    bucket_df.to_csv(bucket_csv, index=False)

    # Vẽ 3 biểu đồ báo cáo
    plot_model_comparison(test_df, fig_dir / "01_test_model_comparison.png")
    plot_feature_importance(feat_imp, fig_dir / "02_top10_feature_importance.png")
    plot_usage_buckets(bucket_df, fig_dir / "03_high_usage_error_analysis.png")

    display_cols = [
        "model",
        "window_days",
        "subset",
        "mae",
        "rmse",
        "bias",
        "high_usage_mae",
        "high_usage_underpred_rate",
        "high_usage_mean_underpred",
    ]
    print("\n=== BẢNG ĐIỂM CHỐT HẠ TRÊN TẬP TEST (THÁNG 9/2026) ===")
    print(test_df[display_cols].to_string(index=False))
    print("\n=== PHÂN TÍCH LỖI THEO MỨC SỬ DỤNG (RQ3 — 90d Tuned Train+Val) ===")
    print(bucket_df.to_string(index=False))
    print(f"\nĐã lưu CSV tại: {test_csv}")
    print(f"Đã lưu 3 biểu đồ tại: {fig_dir}")


if __name__ == "__main__":
    main()
