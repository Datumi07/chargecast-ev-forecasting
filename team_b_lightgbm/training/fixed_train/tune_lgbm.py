"""Chặng 4: Fine-tuning có hệ thống cho mô hình Shared LightGBM (+ com_id).

Quy trình 3 bước có giả thuyết rõ ràng (chỉ dùng Validation tháng 7–8, khóa kín Test tháng 9):
1. Bước 1 (So sánh hàm mục tiêu / objective trên cửa sổ 90 ngày):
   Thử 4 chế độ phạt lỗi bên trong LightGBM:
   - `regression` (L2 / RMSE)
   - `regression_l1` (L1 / MAE)
   - `poisson` (dữ liệu đếm không âm)
   - `tweedie` (dữ liệu đếm không âm có nhiều giá trị 0)
2. Bước 2 (Tinh chỉnh cấu trúc cây, learning_rate thấp & chống học vẹt):
   Chọn 2 hàm mục tiêu tốt nhất từ Bước 1 (theo RMSE), hạ `learning_rate = 0.02`,
   và thử 4 hồ sơ siêu tham số (`num_leaves`, `min_child_samples`, `reg_alpha`,
   `reg_lambda`, `cat_smooth`).
3. Bước 3 (Kiểm chứng đối xứng trên cả cửa sổ 30 ngày và 90 ngày):
   Chạy cấu hình chiến thắng trên cả `window=30` và `window=90`, rồi lưu cấu hình
   tốt nhất vào `reports/modeling/best_lgbm_config.json` cho Chặng 5.

Cách chạy từ thư mục gốc của repo:
    python src/tune_lgbm.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluation import (
    compute_high_usage_threshold,
    evaluate_predictions,
    filter_split,
    load_feature_dataset,
)
from train_lgbm_shared import prepare_features


def run_single_trial(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    predictors: list[str],
    threshold: float,
    window: int,
    trial_name: str,
    params: dict[str, Any],
    stopping_rounds: int = 80,
) -> tuple[list[dict[str, Any]], int]:
    """Chạy 1 cấu hình LightGBM Shared (+ com_id) và trả về bảng chỉ số Validation."""
    t0 = time.time()
    X_train, X_val, cat_cols = prepare_features(
        df_train, df_val, predictors, use_com_id=True
    )
    y_train = df_train["sessions_target"].to_numpy(dtype=float)
    y_val = df_val["sessions_target"].to_numpy(dtype=float)

    model_params = {
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        **params,
    }
    model = lgb.LGBMRegressor(**model_params)

    # Dùng đúng metric theo dõi phù hợp để early stopping dừng chính xác
    eval_metric = "mae" if params.get("objective") == "regression_l1" else "rmse"
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=eval_metric,
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)],
    )

    y_pred = np.clip(model.predict(X_val), 0.0, None)
    best_iter = int(model.best_iteration_)
    elapsed = time.time() - t0

    rows = evaluate_predictions(
        df_split_eligible=df_val,
        y_pred=y_pred,
        high_usage_threshold=threshold,
        model_name=trial_name,
        window=window,
        split_name="validation",
    )
    for r in rows:
        r["objective"] = params.get("objective", "regression")
        r["learning_rate"] = params.get("learning_rate")
        r["num_leaves"] = params.get("num_leaves")
        r["min_child_samples"] = params.get("min_child_samples")
        r["cat_smooth"] = params.get("cat_smooth", 10.0)
        r["best_iteration"] = best_iter
        r["fit_seconds"] = round(elapsed, 1)

    elig_row = next(r for r in rows if r["subset"] == "all_eligible")
    print(
        f"  {trial_name:<34} | {window}d | iter={best_iter:>4} | "
        f"MAE={elig_row['mae']:.4f} | RMSE={elig_row['rmse']:.4f} | "
        f"HiMAE={elig_row['high_usage_mae']:.4f} | "
        f"HiUnder={elig_row['high_usage_underpred_rate']*100:.1f}% ({elapsed:.1f}s)"
    )
    return rows, best_iter


def main() -> None:
    print("=== Chặng 4: Fine-Tuning Shared LightGBM (+ com_id) ===")
    df90, pred90 = load_feature_dataset(90)
    df90_tr = filter_split(df90, "train", eligible_only=True)
    df90_va = filter_split(df90, "validation", eligible_only=True)
    th90 = compute_high_usage_threshold(df90_tr, quantile=0.90)

    all_rows: list[dict[str, Any]] = []
    trial_configs: dict[str, dict[str, Any]] = {}

    # --- BƯỚC 1: So sánh 4 hàm mục tiêu (objective) trên cửa sổ 90 ngày ---
    print("\n[Bước 1] So sánh 4 chế độ phạt lỗi (objective) trên cửa sổ 90 ngày:")
    base_stage1 = {
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
    }
    stage1_objectives: list[tuple[str, dict[str, Any]]] = [
        ("s1_obj_regression_l2", {**base_stage1, "objective": "regression"}),
        ("s1_obj_regression_l1", {**base_stage1, "objective": "regression_l1"}),
        ("s1_obj_poisson", {**base_stage1, "objective": "poisson"}),
        (
            "s1_obj_tweedie_1.3",
            {**base_stage1, "objective": "tweedie", "tweedie_variance_power": 1.3},
        ),
    ]

    stage1_scores: list[tuple[str, float, dict[str, Any]]] = []
    for name, params in stage1_objectives:
        rows, _ = run_single_trial(
            df90_tr, df90_va, pred90, th90, 90, name, params, stopping_rounds=60
        )
        all_rows.extend(rows)
        trial_configs[name] = params
        elig = next(r for r in rows if r["subset"] == "all_eligible")
        stage1_scores.append((name, float(elig["rmse"]), params))

    # Chọn 2 hàm mục tiêu có RMSE Validation thấp nhất để tinh chỉnh sâu ở Bước 2
    stage1_scores.sort(key=lambda x: x[1])
    top2_stage1 = stage1_scores[:2]
    print(
        "\n-> 2 hàm mục tiêu tốt nhất theo RMSE để tinh chỉnh sâu ở Bước 2: "
        + ", ".join(f"{n} (RMSE={s:.4f})" for n, s, _ in top2_stage1)
    )

    # --- BƯỚC 2: Hạ learning_rate=0.02 & tinh chỉnh cấu trúc cây + chống học vẹt ---
    print("\n[Bước 2] Tinh chỉnh siêu tham số (learning_rate=0.02, regularization & cat_smooth):")
    profiles: list[tuple[str, dict[str, Any]]] = [
        (
            "balanced_low_lr",
            {
                "n_estimators": 4000,
                "learning_rate": 0.02,
                "num_leaves": 31,
                "min_child_samples": 50,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
                "cat_smooth": 20.0,
            },
        ),
        (
            "compact_reg",
            {
                "n_estimators": 4000,
                "learning_rate": 0.02,
                "num_leaves": 15,
                "min_child_samples": 100,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "cat_smooth": 30.0,
            },
        ),
        (
            "deeper_smooth",
            {
                "n_estimators": 4000,
                "learning_rate": 0.02,
                "num_leaves": 63,
                "min_child_samples": 100,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.7,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "cat_smooth": 40.0,
            },
        ),
        (
            "high_min_child",
            {
                "n_estimators": 4000,
                "learning_rate": 0.02,
                "num_leaves": 31,
                "min_child_samples": 150,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.7,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "cat_smooth": 30.0,
            },
        ),
    ]

    for obj_name, _, obj_params in top2_stage1:
        obj_tag = obj_name.replace("s1_obj_", "")
        extra_obj = {"objective": obj_params["objective"]}
        if "tweedie_variance_power" in obj_params:
            extra_obj["tweedie_variance_power"] = obj_params["tweedie_variance_power"]

        for prof_name, prof_params in profiles:
            name = f"s2_{obj_tag}_{prof_name}"
            params = {**prof_params, **extra_obj}
            rows, _ = run_single_trial(
                df90_tr, df90_va, pred90, th90, 90, name, params, stopping_rounds=80
            )
            all_rows.extend(rows)
            trial_configs[name] = params

    # Tìm cấu hình tốt nhất trên 90d theo RMSE Validation (all_eligible)
    elig_90_rows = [
        r for r in all_rows if r["window_days"] == 90 and r["subset"] == "all_eligible"
    ]
    best_90_row = min(elig_90_rows, key=lambda r: (r["rmse"], r["mae"]))
    best_name = best_90_row["model"]
    best_params = trial_configs[best_name]

    # --- BƯỚC 3: Kiểm chứng đối xứng cấu hình tốt nhất trên cửa sổ 30 ngày ---
    print(f"\n[Bước 3] Kiểm chứng cấu hình tốt nhất ({best_name}) trên cửa sổ 30 ngày:")
    df30, pred30 = load_feature_dataset(30)
    df30_tr = filter_split(df30, "train", eligible_only=True)
    df30_va = filter_split(df30, "validation", eligible_only=True)
    th30 = compute_high_usage_threshold(df30_tr, quantile=0.90)

    rows_30, best_iter_30 = run_single_trial(
        df30_tr,
        df30_va,
        pred30,
        th30,
        30,
        f"best_tuned_{best_name}",
        best_params,
        stopping_rounds=80,
    )
    all_rows.extend(rows_30)

    # Lưu toàn bộ lịch sử thử nghiệm và cấu hình thắng cuộc
    out_csv = Path("reports/modeling/lgbm_tuning_validation.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_csv, index=False)

    best_cfg_payload = {
        "best_trial_name": best_name,
        "selected_by": "lowest_validation_rmse_all_eligible_90d",
        "params": best_params,
        "validation_90d_all_eligible": best_90_row,
        "validation_30d_best_iteration": best_iter_30,
    }
    out_json = Path("reports/modeling/best_lgbm_config.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best_cfg_payload, f, indent=2, ensure_ascii=False)

    print(f"\n=== Cấu hình chiến thắng trên Validation (90d): {best_name} ===")
    print(
        f"MAE={best_90_row['mae']:.4f} | RMSE={best_90_row['rmse']:.4f} | "
        f"High-usage MAE={best_90_row['high_usage_mae']:.4f} | "
        f"High-usage Underpred={best_90_row['high_usage_underpred_rate']*100:.2f}%"
    )
    print(f"Đã lưu bảng tuning tại: {out_csv}")
    print(f"Đã lưu cấu hình tốt nhất tại: {out_json}")


if __name__ == "__main__":
    main()
