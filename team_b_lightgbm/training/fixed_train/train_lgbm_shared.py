"""Chặng 3A: Huấn luyện mô hình LightGBM dùng chung (Shared Learning).

Mục tiêu:
- Huấn luyện 1 mô hình LightGBM duy nhất trên toàn bộ 1.678 thiết bị sạc (Shared Learning)
  cho cả 2 cửa sổ dữ liệu: 30 ngày và 90 ngày.
- So sánh 2 biến thể:
  1. Chỉ dùng 26 biến dự báo từ `predictors_{window}d.json`.
  2. Thêm `com_id` dạng biến phân loại (categorical) được khóa danh mục chỉ trên tập Train
     để chống rò rỉ dữ liệu (data leakage).
- Sử dụng Early Stopping trên tập Validation (tháng 7–8) để tránh học vẹt (overfitting).
- Tuyệt đối KHÔNG chạm vào tập Test (tháng 9) trong giai đoạn này.

Cách chạy từ thư mục gốc của repo:
    python team_b_lightgbm/training/fixed_train/train_lgbm_shared.py
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluation import (
    compute_high_usage_threshold,
    evaluate_predictions,
    filter_split,
    load_feature_dataset,
)


def prepare_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    predictors: list[str],
    use_com_id: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Chuẩn bị ma trận đặc trưng X_train, X_val không rò rỉ dữ liệu.

    Lưu ý:
    - KHÔNG điền 0 vào giá trị NaN (LightGBM tự học nhánh tối ưu cho NaN).
    - Nếu dùng `com_id`, danh sách categories chỉ được lấy từ tập Train.
    """
    feature_cols = list(predictors)
    X_train = df_train[feature_cols].copy()
    X_val = df_val[feature_cols].copy()
    categorical_cols: list[str] = []

    if use_com_id:
        # Khóa danh mục com_id chỉ trên tập Train để tránh data leakage
        train_categories = pd.Index(df_train["com_id"].dropna().unique())
        X_train["com_id"] = pd.Categorical(df_train["com_id"], categories=train_categories)
        X_val["com_id"] = pd.Categorical(df_val["com_id"], categories=train_categories)
        categorical_cols.append("com_id")

    return X_train, X_val, categorical_cols


def train_and_evaluate_shared(
    window: int,
    use_com_id: bool = False,
    high_usage_quantile: float = 0.90,
) -> tuple[list[dict], pd.DataFrame]:
    """Huấn luyện 1 mô hình Shared LightGBM cơ bản và đánh giá trên Validation."""
    df, predictors = load_feature_dataset(window)

    # 1. Lọc tập Train (tháng 2–6) và Validation (tháng 7–8) theo chuẩn label_eligible
    df_train = filter_split(df, "train", eligible_only=True)
    df_val = filter_split(df, "validation", eligible_only=True)

    # 2. Tính ngưỡng ngày cao điểm CHỈ trên tập Train (top 10% -> 4.0 phiên)
    threshold = compute_high_usage_threshold(df_train, quantile=high_usage_quantile)

    X_train, X_val, cat_cols = prepare_features(
        df_train, df_val, predictors, use_com_id=use_com_id
    )
    y_train = df_train["sessions_target"].to_numpy(dtype=float)
    y_val = df_val["sessions_target"].to_numpy(dtype=float)

    # 3. Khởi tạo LightGBM với cấu hình cơ bản (trước khi fine-tune ở Chặng 4)
    model = lgb.LGBMRegressor(
        objective="regression",   # Tối ưu bình phương sai số (L2 / RMSE)
        n_estimators=2000,        # Số cây tối đa (sẽ dừng sớm nhờ early stopping)
        learning_rate=0.05,       # Bước học nhỏ để mỗi cây sửa sai từ từ
        num_leaves=31,            # Độ phức tạp của mỗi cây (mặc định chuẩn của LightGBM)
        min_child_samples=20,     # Số mẫu tối thiểu trong 1 lá để tránh học vẹt
        subsample=0.8,            # Lấy ngẫu nhiên 80% dữ liệu mỗi vòng để giảm overfit
        subsample_freq=1,
        colsample_bytree=0.8,     # Lấy ngẫu nhiên 80% cột cho mỗi cây
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # 4. Huấn luyện kèm Early Stopping: nếu 50 cây liên tiếp lỗi Validation không giảm thì dừng
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # 5. Dự báo trên Validation và chặn dưới ở 0 (số phiên sạc không thể âm)
    y_pred = np.clip(model.predict(X_val), 0.0, None)

    variant = "with_com_id" if use_com_id else "26feats"
    model_name = f"lgbm_shared_{variant}"

    rows = evaluate_predictions(
        df_split_eligible=df_val,
        y_pred=y_pred,
        high_usage_threshold=threshold,
        model_name=model_name,
        window=window,
        split_name="validation",
    )
    for r in rows:
        r["best_iteration"] = int(model.best_iteration_)

    # 6. Trích xuất mức độ quan trọng của đặc trưng (theo tổng mức giảm lỗi - gain)
    gain_importance = model.booster_.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame(
        {
            "feature": X_train.columns,
            "gain": gain_importance,
            "gain_pct": 100.0 * gain_importance / max(gain_importance.sum(), 1e-12),
        }
    ).sort_values("gain", ascending=False)

    print(
        f"\n[{model_name} | window={window}d] "
        f"Dừng sớm tại cây số {model.best_iteration_} / 2000"
    )
    print("Top 8 đặc trưng quan trọng nhất (theo % Gain):")
    print(feat_imp.head(8).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return rows, feat_imp


def main() -> None:
    all_rows: list[dict] = []

    for window in (30, 90):
        for use_com_id in (False, True):
            rows, _ = train_and_evaluate_shared(window=window, use_com_id=use_com_id)
            all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    out_path = Path("reports/modeling/lgbm_shared_validation.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    display_cols = [
        "model",
        "window_days",
        "subset",
        "best_iteration",
        "mae",
        "rmse",
        "high_usage_mae",
        "high_usage_underpred_rate",
        "high_usage_mean_underpred",
    ]
    print("\n=== Kết quả LightGBM Shared (Validation) ===")
    print(results_df[display_cols].to_string(index=False))
    print(f"\nĐã lưu kết quả tại: {out_path}")


if __name__ == "__main__":
    main()
