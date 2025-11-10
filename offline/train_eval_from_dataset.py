# offline/train_eval_from_dataset.py
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump

DATA_DIR = "data"
DATASET_PATH = os.path.join(DATA_DIR, "dataset_generated.csv")
RANDOM_STATE = 42
N_FOLDS = 5


# 우리가 예측하고 싶은 타깃들 (필요한 것만 골라서)
TARGETS = [
    "depression",
    "happines",
    "STRESS",
    "ATTACK",
]


def extract_linear_formula(model, feature_names):
    """
    Lasso 같은 선형 모델에서 가중치 + 절편을
    사람이 읽을 수 있는 수식 문자열로 변환.
    """
    coef = model.coef_
    intercept = float(model.intercept_)
    terms = []
    for w, name in zip(coef, feature_names):
        if abs(w) < 1e-6:
            continue
        terms.append(f"{w:.4f}*{name}")
    if not terms:
        expr = f"{intercept:.4f}"
    else:
        expr = " + ".join(terms) + f" + {intercept:.4f}"
    return "output = " + expr


def main():
    df = pd.read_csv(DATASET_PATH)

    label_cols = [
        "brainfatigue",
        "concentrate",
        "LIFEPOWER",
        "ATTACK",
        "STRESS",
        "UNSTABLE",
        "SUSPICION",
        "BALANCE",
        "CHARM",
        "ENERGY",
        "SELFCONTROL",
        "HYSTERIA",
        "depression",
        "happines",
    ]

    drop_cols = ["video_id"] + label_cols
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_all = df[feature_cols].values
    n_samples = len(df)
    n_splits = max(2, min(N_FOLDS, n_samples))

    print(f"[INFO] n_samples = {n_samples}, n_splits = {n_splits}")
    print(f"[INFO] feature_cols = {feature_cols}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    for target in TARGETS:
        y_all = df[target].values
        fold_mae = []
        fold_r2 = []

        for train_idx, test_idx in kf.split(X_all):
            X_tr, X_te = X_all[train_idx], X_all[test_idx]
            y_tr, y_te = y_all[train_idx], y_all[test_idx]

            model = Lasso(alpha=0.001, max_iter=5000)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            mae = mean_absolute_error(y_te, y_pred)
            r2 = r2_score(y_te, y_pred)
            fold_mae.append(mae)
            fold_r2.append(r2)

        avg_mae = float(np.mean(fold_mae))
        avg_r2 = float(np.mean(fold_r2))

        # 전체 데이터로 최종 모델 적합
        final_model = Lasso(alpha=0.001, max_iter=5000)
        final_model.fit(X_all, y_all)

        os.makedirs("modeling/checkpoints", exist_ok=True)
        dump(final_model, f"modeling/checkpoints/{target}_lasso.joblib")

        formula = extract_linear_formula(final_model, feature_cols)

        results[target] = {
            "cv_mae": avg_mae,
            "cv_r2": avg_r2,
            "formula": formula,
        }

    print("=== CROSS-VALIDATION RESULTS ===")
    for target, info in results.items():
        print(f"\nTarget: {target}")
        print(f"  MAE: {info['cv_mae']:.4f}")
        print(f"  R^2: {info['cv_r2']:.4f}")
        print("  Formula:")
        print("   ", info["formula"])


if __name__ == "__main__":
    main()
