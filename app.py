import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Manufacturing Efficiency", layout="wide")


@st.cache_resource
def load_artifacts(model_path: str):
    obj = joblib.load(model_path)
    return obj["model"], obj.get("classes"), obj.get("best_model_name")


@st.cache_data
def load_data(csv_path: str, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=nrows)
    if "Time" in df.columns and "Timestamp" not in df.columns:
        df = df.rename(columns={"Time": "Timestamp"})
    df["_dt"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Timestamp"].astype(str),
        errors="coerce",
        dayfirst=True,
    )
    df = df.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)
    return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW"]:
        if col in out.columns:
            out[f"{col}_roll_std_15"] = (
                out.groupby("Machine_ID")[col]
                .rolling(window=15, min_periods=5)
                .std()
                .reset_index(level=0, drop=True)
            )

    if "Production_Speed_units_per_hr" in out.columns and "Power_Consumption_kW" in out.columns:
        out["Units_per_kW"] = out["Production_Speed_units_per_hr"] / (out["Power_Consumption_kW"].replace(0, np.nan))

    if "Error_Rate_%" in out.columns and "Production_Speed_units_per_hr" in out.columns:
        out["ErrorRate_per_Unit"] = out["Error_Rate_%"] / (out["Production_Speed_units_per_hr"].replace(0, np.nan))

    if "Quality_Control_Defect_Rate_%" in out.columns and "Production_Speed_units_per_hr" in out.columns:
        out["DefectRate_per_Unit"] = out["Quality_Control_Defect_Rate_%"] / (
            out["Production_Speed_units_per_hr"].replace(0, np.nan)
        )

    if "Network_Latency_ms" in out.columns and "Packet_Loss_%" in out.columns:
        lat = out["Network_Latency_ms"].clip(lower=0)
        loss = out["Packet_Loss_%"].clip(lower=0)
        lat_n = lat / (lat.quantile(0.95) if lat.quantile(0.95) > 0 else 1.0)
        loss_n = loss / (loss.quantile(0.95) if loss.quantile(0.95) > 0 else 1.0)
        out["Network_Reliability_Score"] = 1.0 - (0.6 * lat_n + 0.4 * loss_n)

    out["Hour"] = out["_dt"].dt.hour
    out["DayOfWeek"] = out["_dt"].dt.dayofweek

    return out


def compute_network_reliability(df: pd.DataFrame) -> pd.Series:
    lat = df["Network_Latency_ms"].clip(lower=0)
    loss = df["Packet_Loss_%"].clip(lower=0)
    lat_n = lat / (lat.quantile(0.95) if lat.quantile(0.95) > 0 else 1.0)
    loss_n = loss / (loss.quantile(0.95) if loss.quantile(0.95) > 0 else 1.0)
    return 1.0 - (0.6 * lat_n + 0.4 * loss_n)


def main():
    st.title("AI-Based Manufacturing Efficiency Classification")

    with st.sidebar:
        st.header("Inputs")
        csv_path = st.text_input("CSV path", value="Thales_Group_Manufacturing.csv")
        model_path = st.text_input("Model path", value="model.joblib")
        metrics_path = st.text_input("Metrics path", value="metrics.json")

        st.divider()
        st.subheader("Filters")
        max_rows = st.number_input("Rows to load (for speed)", min_value=2000, max_value=100000, value=20000, step=2000)

    try:
        df = load_data(csv_path, nrows=int(max_rows))
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()

    # User controls
    with st.sidebar:
        machines = sorted(df["Machine_ID"].dropna().unique().tolist())
        selected_machines = st.multiselect("Machine selector", machines, default=machines[:3] if len(machines) >= 3 else machines)

        modes = sorted(df["Operation_Mode"].dropna().unique().tolist())
        selected_mode = st.selectbox("Operation mode dropdown", ["All"] + modes)

        if df["_dt"].notna().any():
            min_dt = df["_dt"].min()
            max_dt = df["_dt"].max()
            time_window = st.slider("Time window filter", min_value=min_dt.to_pydatetime(), max_value=max_dt.to_pydatetime(), value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()))
        else:
            time_window = None

        st.divider()
        st.subheader("Metric sensitivity sliders")
        latency_weight = st.slider("Latency weight", 0.0, 1.0, 0.6, 0.05)
        loss_weight = 1.0 - latency_weight

    dff = df.copy()
    if selected_machines:
        dff = dff[dff["Machine_ID"].isin(selected_machines)]
    if selected_mode != "All":
        dff = dff[dff["Operation_Mode"] == selected_mode]
    if time_window is not None:
        start, end = time_window
        dff = dff[(dff["_dt"] >= pd.to_datetime(start)) & (dff["_dt"] <= pd.to_datetime(end))]

    dff = add_feature_engineering(dff)

    # Operational Monitoring View: network vs sensor
    st.subheader("Operational Monitoring View")
    cols = st.columns(4)

    if {"Network_Latency_ms", "Packet_Loss_%"}.issubset(dff.columns) and len(dff) > 0:
        lat = dff["Network_Latency_ms"].clip(lower=0)
        loss = dff["Packet_Loss_%"].clip(lower=0)
        lat_n = lat / (lat.quantile(0.95) if lat.quantile(0.95) > 0 else 1.0)
        loss_n = loss / (loss.quantile(0.95) if loss.quantile(0.95) > 0 else 1.0)
        dff["Network_Reliability_Score_UI"] = 1.0 - (latency_weight * lat_n + loss_weight * loss_n)

    with cols[0]:
        st.metric("Rows", f"{len(dff):,}")
    with cols[1]:
        st.metric("Avg Temp (C)", f"{dff['Temperature_C'].mean():.2f}" if "Temperature_C" in dff.columns else "N/A")
    with cols[2]:
        st.metric("Avg Latency (ms)", f"{dff['Network_Latency_ms'].mean():.2f}" if "Network_Latency_ms" in dff.columns else "N/A")
    with cols[3]:
        if "Network_Reliability_Score_UI" in dff.columns:
            st.metric("Avg Reliability", f"{dff['Network_Reliability_Score_UI'].mean():.3f}")
        elif "Network_Reliability_Score" in dff.columns:
            st.metric("Avg Reliability", f"{dff['Network_Reliability_Score'].mean():.3f}")
        else:
            st.metric("Avg Reliability", "N/A")

    if len(dff) == 0:
        st.warning("No data after filters.")
        st.stop()

    st.divider()

    # Load model
    try:
        model, classes, best_model_name = load_artifacts(model_path)
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

    # Efficiency Prediction Dashboard
    st.subheader("Efficiency Prediction Dashboard")

    # Predict on latest N rows for a pseudo real-time view
    latest_n = min(2000, len(dff))
    latest = dff.tail(latest_n).copy()

    X_latest = latest.drop(columns=[c for c in ["Efficiency_Status"] if c in latest.columns])

    try:
        prob = model.predict_proba(X_latest)
        pred_idx = np.argmax(prob, axis=1)
        pred = model.classes_[pred_idx]
        conf = prob[np.arange(len(pred_idx)), pred_idx]
        latest["Predicted_Efficiency"] = pred
        latest["Prediction_Confidence"] = conf
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write(f"**Model in use:** {best_model_name}")
        st.dataframe(latest[["_dt", "Machine_ID", "Operation_Mode", "Predicted_Efficiency", "Prediction_Confidence"]].tail(30), use_container_width=True)

    with c2:
        fig = px.histogram(latest, x="Prediction_Confidence", nbins=20, title="Confidence score visualization")
        st.plotly_chart(fig, use_container_width=True)

    # Machine-Level Insights
    st.subheader("Machine-Level Insights")
    g = (
        latest.groupby(["Machine_ID", "Predicted_Efficiency"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    fig2 = px.bar(g, x="Machine_ID", y="count", color="Predicted_Efficiency", barmode="group", title="Efficiency trends per machine")
    st.plotly_chart(fig2, use_container_width=True)

    # Explainability Panel (simple feature importance when available)
    st.subheader("Explainability Panel")
    st.write("Feature importance charts (model-dependent).")

    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
    pre = model.named_steps.get("pre") if hasattr(model, "named_steps") else None

    if clf is not None and pre is not None and hasattr(clf, "feature_importances_"):
        try:
            feature_names = pre.get_feature_names_out()
            imp = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
            imp = imp.sort_values("importance", ascending=False).head(20)
            fig3 = px.bar(imp, x="importance", y="feature", orientation="h", title="Top drivers of efficiency status")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.info(f"Could not compute feature importance: {e}")
    elif clf is not None and pre is not None and hasattr(clf, "coef_"):
        try:
            feature_names = pre.get_feature_names_out()
            coef = np.asarray(clf.coef_)
            coef_abs = np.mean(np.abs(coef), axis=0)
            imp = pd.DataFrame({"feature": feature_names, "importance": coef_abs})
            imp = imp.sort_values("importance", ascending=False).head(20)
            fig3 = px.bar(imp, x="importance", y="feature", orientation="h", title="Top drivers of efficiency status")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.info(f"Could not compute coefficient importance: {e}")
    else:
        st.info("Feature importance not available for this model type in this dashboard.")

    enable_shap = st.checkbox("Explain why a sample is classified as Low / Medium / High (SHAP)", value=False)
    if enable_shap and clf is not None and pre is not None:
        try:
            import shap

            row_idx = st.slider("Sample row (from latest window)", min_value=0, max_value=len(latest) - 1, value=len(latest) - 1)
            x_row = X_latest.iloc[[row_idx]]
            x_row_t = pre.transform(x_row)

            bg_n = min(250, len(X_latest))
            bg = X_latest.sample(bg_n, random_state=42) if len(X_latest) > bg_n else X_latest
            bg_t = pre.transform(bg)

            if hasattr(clf, "predict_proba"):
                pred_label = str(latest.iloc[row_idx]["Predicted_Efficiency"])
                class_names = [str(c) for c in getattr(model, "classes_", [])]
                class_i = class_names.index(pred_label) if pred_label in class_names else 0
            else:
                class_i = 0

            explainer = shap.TreeExplainer(clf, bg_t) if hasattr(clf, "feature_importances_") else shap.Explainer(clf, bg_t)
            sv = explainer(x_row_t)

            feature_names = pre.get_feature_names_out()

            if hasattr(sv, "values"):
                values = sv.values
                if isinstance(values, list):
                    vals = np.asarray(values[class_i]).ravel()
                else:
                    arr = np.asarray(values)
                    vals = arr[0, :, class_i] if arr.ndim == 3 else arr[0]
            else:
                vals = np.zeros(len(feature_names))

            local = pd.DataFrame({"feature": feature_names, "shap_value": vals})
            local["abs"] = np.abs(local["shap_value"])
            local = local.sort_values("abs", ascending=False).head(15)
            fig_local = px.bar(local.iloc[::-1], x="shap_value", y="feature", orientation="h", title="Local explanation (top contributions)")
            st.plotly_chart(fig_local, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

    # Metrics readout
    st.subheader("Model Development Summary")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.json(metrics)
    except Exception:
        st.info("metrics.json not found yet. Run training first to generate it.")


if __name__ == "__main__":
    main()
