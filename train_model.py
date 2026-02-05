import argparse
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


class LabelEncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params

    def fit(self, X, y):
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost is not installed")

        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(pd.Series(y).astype(str))

        self._model = XGBClassifier(**self.xgb_params)
        self._model.fit(X, y_enc)

        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        pred_enc = self._model.predict(X)
        pred_enc = np.asarray(pred_enc, dtype=int)
        return self._le.inverse_transform(pred_enc)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


TARGET_COL = "Efficiency_Status"
DATE_COL = "Date"
TIME_COL = "Timestamp"


@dataclass
class TrainArtifacts:
    best_model_name: str
    best_model: object
    label_classes: list[str]
    metrics_by_model: dict


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if TIME_COL not in df.columns and "Time" in df.columns:
        df = df.rename(columns={"Time": TIME_COL})

    if DATE_COL not in df.columns:
        raise ValueError(f"Expected '{DATE_COL}' column")
    if TIME_COL not in df.columns:
        raise ValueError(f"Expected '{TIME_COL}' column")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target '{TARGET_COL}' column")

    df["_dt"] = pd.to_datetime(
        df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str),
        errors="coerce",
        dayfirst=True,
    )
    df = df.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)

    return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Sensor stability indicators (rolling std by machine)
    for col in ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW"]:
        if col in out.columns:
            out[f"{col}_roll_std_15"] = (
                out.groupby("Machine_ID")[col]
                .rolling(window=15, min_periods=5)
                .std()
                .reset_index(level=0, drop=True)
            )

    # Energy efficiency ratios
    if "Production_Speed_units_per_hr" in out.columns and "Power_Consumption_kW" in out.columns:
        out["Units_per_kW"] = out["Production_Speed_units_per_hr"] / (out["Power_Consumption_kW"].replace(0, np.nan))

    # Error-to-output ratios
    if "Error_Rate_%" in out.columns and "Production_Speed_units_per_hr" in out.columns:
        out["ErrorRate_per_Unit"] = out["Error_Rate_%"] / (out["Production_Speed_units_per_hr"].replace(0, np.nan))

    if "Quality_Control_Defect_Rate_%" in out.columns and "Production_Speed_units_per_hr" in out.columns:
        out["DefectRate_per_Unit"] = out["Quality_Control_Defect_Rate_%"] / (
            out["Production_Speed_units_per_hr"].replace(0, np.nan)
        )

    # Network reliability score (higher is better)
    if "Network_Latency_ms" in out.columns and "Packet_Loss_%" in out.columns:
        # Normalize to 0..1 using robust-ish clipping then invert
        lat = out["Network_Latency_ms"].clip(lower=0)
        loss = out["Packet_Loss_%"].clip(lower=0)
        lat_n = lat / (lat.quantile(0.95) if lat.quantile(0.95) > 0 else 1.0)
        loss_n = loss / (loss.quantile(0.95) if loss.quantile(0.95) > 0 else 1.0)
        out["Network_Reliability_Score"] = 1.0 - (0.6 * lat_n + 0.4 * loss_n)

    # Time-derived
    out["Hour"] = out["_dt"].dt.hour
    out["DayOfWeek"] = out["_dt"].dt.dayofweek

    return out


def build_preprocess_pipeline(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, DATE_COL, TIME_COL, "_dt"}]

    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    return pre, num_cols, cat_cols


def time_series_cv_fit(model, X, y, n_splits: int = 5):
    tss = TimeSeriesSplit(n_splits=n_splits)

    accs = []
    last_fold_report = None
    last_cm = None

    for train_idx, test_idx in tss.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, pred))
        last_fold_report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        last_cm = confusion_matrix(y_test, pred)

    return {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "last_fold_report": last_fold_report,
        "last_fold_confusion_matrix": last_cm.tolist() if last_cm is not None else None,
    }


def train_all_models(df: pd.DataFrame, class_weight: str | None = "balanced") -> TrainArtifacts:
    df = add_feature_engineering(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(str)

    pre, _, _ = build_preprocess_pipeline(df)

    models = {}

    models["logreg"] = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=None,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    models["rf"] = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight=class_weight,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    if HAS_XGBOOST:
        models["xgb"] = Pipeline(
            steps=[
                ("pre", pre),
                (
                    "clf",
                    LabelEncodedXGBClassifier(
                        n_estimators=400,
                        max_depth=6,
                        learning_rate=0.06,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
                    ),
                ),
            ]
        )

    metrics = {}
    best_name = None
    best_score = -1

    for name, pipe in models.items():
        metrics[name] = time_series_cv_fit(pipe, X, y, n_splits=5)
        score = metrics[name]["cv_accuracy_mean"]
        if score > best_score:
            best_score = score
            best_name = name

    best_model = models[best_name]
    best_model.fit(X, y)

    classes = sorted(y.unique().tolist())

    return TrainArtifacts(
        best_model_name=best_name,
        best_model=best_model,
        label_classes=classes,
        metrics_by_model=metrics,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Thales_Group_Manufacturing.csv")
    ap.add_argument("--out_model", default="model.joblib")
    ap.add_argument("--out_metrics", default="metrics.json")
    args = ap.parse_args()

    df = load_data(args.csv)

    # Quick imbalance check
    vc = df[TARGET_COL].value_counts(dropna=False)
    imbalance_ratio = float(vc.max() / max(vc.min(), 1))
    class_weight = "balanced" if imbalance_ratio >= 1.5 else None

    artifacts = train_all_models(df, class_weight=class_weight)

    joblib.dump(
        {
            "model": artifacts.best_model,
            "classes": artifacts.label_classes,
            "best_model_name": artifacts.best_model_name,
        },
        args.out_model,
    )

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model_name": artifacts.best_model_name,
                "metrics_by_model": artifacts.metrics_by_model,
                "class_distribution": vc.to_dict(),
                "imbalance_ratio": imbalance_ratio,
                "class_weight": class_weight,
            },
            f,
            indent=2,
        )

    print(f"Saved model to {args.out_model}")
    print(f"Saved metrics to {args.out_metrics}")
    print(f"Best model: {artifacts.best_model_name} (cv acc={artifacts.metrics_by_model[artifacts.best_model_name]['cv_accuracy_mean']:.4f})")


if __name__ == "__main__":
    main()
