

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="🛡️ Fraud Operations Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------
# Custom Styling
# ---------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .subtitle {
        color: #6c757d;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛡️ Fraud Operations Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive dashboard for credit card fraud detection and explainable AI.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "dashboard_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# ---------------------------------------------------
# Risk Tier Assignment
# ---------------------------------------------------
def assign_risk(prob):
    if prob >= 0.75:
        return "Critical Risk"
    elif prob >= 0.40:
        return "Suspicious"
    else:
        return "Clear"

# ---------------------------------------------------
# Align Features to Model
# ---------------------------------------------------
def prepare_features(row_df):
    drop_cols = [
        col for col in [
            "TransactionID",
            "ActualFraud",
            "FraudProbability",
            "RiskTier"
        ]
        if col in row_df.columns
    ]

    X = row_df.drop(columns=drop_cols).copy()

    if model is not None and hasattr(model, "feature_name_"):
        feature_names = list(model.feature_name_)

        # Add missing columns
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0

        # Keep only required columns in correct order
        X = X[feature_names]

    return X

# ---------------------------------------------------
# Safe Prediction
# ---------------------------------------------------
def predict_probability(row_df):
    if model is None:
        return None

    try:
        X = prepare_features(row_df)
        prob = model.predict_proba(X)[:, 1][0]
        return float(prob)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ---------------------------------------------------
# Load Data and Model
# ---------------------------------------------------
df = load_data()
model = load_model()

# Create RiskTier if missing
if "RiskTier" not in df.columns and "FraudProbability" in df.columns:
    df["RiskTier"] = df["FraudProbability"].apply(assign_risk)

# Ensure TransactionID numeric
if "TransactionID" in df.columns:
    df["TransactionID"] = pd.to_numeric(df["TransactionID"], errors="coerce")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Transaction Explorer", "SHAP Explainer"]
)

# Risk Tier Filter
if "RiskTier" in df.columns:
    tiers = sorted(df["RiskTier"].dropna().unique())

    selected_tiers = st.sidebar.multiselect(
        "Filter by Risk Tier",
        options=tiers,
        default=tiers
    )

    filtered_df = df[df["RiskTier"].isin(selected_tiers)].copy()
else:
    filtered_df = df.copy()

# ---------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------
if page == "Overview":
    st.header("📊 Overview")

    total_transactions = len(filtered_df)

    total_fraud = (
        int(filtered_df["ActualFraud"].sum())
        if "ActualFraud" in filtered_df.columns
        else 0
    )

    fraud_rate = (
        total_fraud / total_transactions * 100
        if total_transactions > 0
        else 0
    )

    avg_fraud_amount = 0.0
    if (
        "TransactionAmt" in filtered_df.columns
        and "ActualFraud" in filtered_df.columns
        and total_fraud > 0
    ):
        avg_fraud_amount = (
            filtered_df.loc[
                filtered_df["ActualFraud"] == 1,
                "TransactionAmt"
            ].mean()
        )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Cases", f"{total_fraud:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("Avg Fraud Amount", f"${avg_fraud_amount:,.2f}")

    # Risk Tier Distribution
    if "RiskTier" in filtered_df.columns:
        risk_counts = filtered_df["RiskTier"].value_counts().reset_index()
        risk_counts.columns = ["Risk Tier", "Count"]

        fig = px.bar(
            risk_counts,
            x="Risk Tier",
            y="Count",
            color="Risk Tier",
            title="Transactions by Risk Tier"
        )
        st.plotly_chart(fig, use_container_width=True)


        if "RiskTier" in filtered_df.columns:
            pie_data = filtered_df["RiskTier"].value_counts().reset_index()
            pie_data.columns = ["Risk Tier", "Count"]

            fig_pie = px.pie(
                pie_data,
                names="Risk Tier",
                values="Count",
                hole=0.55,
                title="Risk Tier Distribution"
            )

            st.plotly_chart(fig_pie, use_container_width=True)

    # Fraud Rate by Hour
    if (
        "HourOfDay" in filtered_df.columns
        and "ActualFraud" in filtered_df.columns
    ):
        hourly = (
            filtered_df
            .groupby("HourOfDay")["ActualFraud"]
            .mean()
            .reset_index()
        )
        hourly["ActualFraud"] *= 100

        fig_hour = px.line(
            hourly,
            x="HourOfDay",
            y="ActualFraud",
            markers=True,
            title="Fraud Rate by Hour of Day"
        )
        fig_hour.update_yaxes(title="Fraud Rate (%)")
        st.plotly_chart(fig_hour, use_container_width=True)

        

# ---------------------------------------------------
# PAGE 2: TRANSACTION EXPLORER
# ---------------------------------------------------
elif page == "Transaction Explorer":
    st.header("🔍 Transaction Explorer")

    # Show first 1000 filtered rows
    st.dataframe(
        filtered_df.head(1000),
        use_container_width=True
    )

    # ===== TRANSACTION EXPLORER LOOKUP =====
    if "TransactionID" in filtered_df.columns:
        tx_input = st.text_input("Enter TransactionID")

        if tx_input:
            try:
                tx_id = float(tx_input)

                # Find nearest TransactionID
                temp_df = filtered_df.copy()
                temp_df["distance"] = (
                    temp_df["TransactionID"] - tx_id
                ).abs()

                result = temp_df.nsmallest(1, "distance")

                # Check if closest match is within tolerance
                if result.empty or result.iloc[0]["distance"] > 1e-3:
                    st.warning("TransactionID not found.")
                else:
                    # Get matching row
                    row = result.drop(
                        columns=["distance"]
                    ).iloc[[0]]

                    # Show transaction details
                    st.subheader("Transaction Details")
                    st.dataframe(
                        row,
                        use_container_width=True
                    )

                    # Use stored probability if available,
                    # otherwise generate prediction
                    if "FraudProbability" in row.columns:
                        prob = float(
                            row.iloc[0]["FraudProbability"]
                        )
                    else:
                        prob = predict_probability(row)

                    if prob is not None:
                        risk = assign_risk(prob)

                        # Display metrics
                        col1, col2 = st.columns(2)
                        col1.metric(
                            "Fraud Probability",
                            f"{prob:.4f}"
                        )
                        col2.metric(
                            "Risk Tier",
                            risk
                        )

                        # Risk message
                        if risk == "Critical Risk":
                            st.error(
                                "🚨 Critical Risk: "
                                "This transaction should be blocked immediately."
                            )
                        elif risk == "Suspicious":
                            st.warning(
                                "⚠️ Suspicious: "
                                "Manual review is recommended."
                            )
                        else:
                            st.success(
                                "✅ Clear: "
                                "This transaction appears legitimate."
                            )

            except ValueError:
                st.warning(
                    "Please enter a valid numeric TransactionID."
                )

# ---------------------------------------------------
# PAGE 3: SHAP EXPLAINER
# ---------------------------------------------------
elif page == "SHAP Explainer":
    st.header("🧠 SHAP Explainer")

    if not SHAP_AVAILABLE:
        st.warning("SHAP is not installed. Add 'shap' to requirements.txt.")
        st.stop()

    if model is None:
        st.warning("Model file not found.")
        st.stop()

    if "TransactionID" not in df.columns:
        st.warning("TransactionID column not available.")
        st.stop()

    # ===== SHAP EXPLAINER LOOKUP =====
    tx_input = st.text_input("Enter TransactionID to Explain")

    if tx_input:
        try:
            tx_id = float(tx_input)

            # Find nearest TransactionID (works with scaled float IDs)
            temp_df = df.copy()
            temp_df["distance"] = (
                temp_df["TransactionID"] - tx_id
            ).abs()

            nearest = temp_df.nsmallest(1, "distance")

            # Check if match is close enough
            if nearest.empty or nearest.iloc[0]["distance"] > 1e-3:
                st.warning("TransactionID not found.")
            else:
                # Get matching row
                row = nearest.drop(columns=["distance"]).iloc[[0]]

                # Prepare features
                X_row = prepare_features(row)

                # Predict probability
                prob = predict_probability(row)

                if prob is not None:
                    risk = assign_risk(prob)

                    # Display metrics
                    col1, col2 = st.columns(2)
                    col1.metric("Fraud Probability", f"{prob:.4f}")
                    col2.metric("Risk Tier", risk)

                    # Plain-English explanation
                    if risk == "Critical Risk":
                        st.error(
                            "🚨 Very high fraud probability. "
                            "The transaction should be blocked immediately."
                        )
                    elif risk == "Suspicious":
                        st.warning(
                            "⚠️ Elevated risk detected. "
                            "Manual review is recommended."
                        )
                    else:
                        st.success(
                            "✅ Low risk. The transaction appears legitimate."
                        )

                # -------------------------------
                # SHAP Explanation
                # -------------------------------
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_row)

                    # Handle different SHAP output formats
                    if isinstance(shap_values, list):
                        values = np.array(shap_values[1][0])
                    else:
                        values = np.array(shap_values)

                        if values.ndim == 3:
                            values = values[0, :, 1]
                        elif values.ndim == 2:
                            values = values[0]
                        else:
                            values = values.flatten()

                    # Ensure length matches features
                    values = values[:len(X_row.columns)]

                    # Create importance DataFrame
                    importance = pd.DataFrame({
                        "Feature": X_row.columns,
                        "SHAP Value": values
                    })

                    importance["Absolute SHAP"] = (
                        importance["SHAP Value"].abs()
                    )

                    # Top 10 most important features
                    importance = (
                        importance
                        .sort_values(
                            "Absolute SHAP",
                            ascending=False
                        )
                        .head(10)
                    )

                    # Show table
                    st.subheader("Top Contributing Features")
                    st.dataframe(
                        importance[["Feature", "SHAP Value"]],
                        use_container_width=True
                    )

                    # Key Fraud Signals
                    st.subheader("Key Fraud Signals")

                    top3 = importance.head(3)

                    for _, feature_row in top3.iterrows():
                        direction = (
                            "increases"
                            if feature_row["SHAP Value"] > 0
                            else "decreases"
                        )

                        st.write(
                            f"• **{feature_row['Feature']}** "
                            f"{direction} fraud risk "
                            f"({feature_row['SHAP Value']:.4f})"
                        )

                    # Plotly bar chart
                    fig = px.bar(
                        importance.sort_values("SHAP Value"),
                        x="SHAP Value",
                        y="Feature",
                        orientation="h",
                        title="SHAP Feature Contributions",
                        color="SHAP Value"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                except Exception as e:
                    st.warning(
                        f"Unable to generate SHAP explanation: {e}"
                    )

        except ValueError:
            st.warning(
                "Please enter a valid numeric TransactionID."
            )