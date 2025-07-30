import streamlit as st
import pandas as pd
import joblib
import shap

# Load model and feature list
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📉 Telco Customer Churn Predictor")

st.markdown(
    "Provide customer details below to predict whether they are likely to churn."
)

# ===== Define Inputs =====

gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.selectbox("Has Partner?", ["No", "Yes"])
Dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Has Phone Service?", ["No", "Yes"])
MultipleLines = st.selectbox(
    "Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox(
    "Online Security", ["No internet service", "No", "Yes"])
OnlineBackup = st.selectbox(
    "Online Backup", ["No internet service", "No", "Yes"])
DeviceProtection = st.selectbox(
    "Device Protection", ["No internet service", "No", "Yes"])
TechSupport = st.selectbox(
    "Tech Support", ["No internet service", "No", "Yes"])
StreamingTV = st.selectbox(
    "Streaming TV", ["No internet service", "No", "Yes"])
StreamingMovies = st.selectbox(
    "Streaming Movies", ["No internet service", "No", "Yes"])
Contract = st.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing?", ["No", "Yes"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
TotalCharges = st.slider("Total Charges", 0.0, 10000.0, 2500.0)


# ===== Build Input DataFrame =====


def preprocess_input():
    df = pd.DataFrame(
        {
            "gender": [1 if gender == "Male" else 0],
            "SeniorCitizen": [1 if SeniorCitizen == "Yes" else 0],
            "Partner": [1 if Partner == "Yes" else 0],
            "Dependents": [1 if Dependents == "Yes" else 0],
            "tenure": [tenure],
            "PhoneService": [1 if PhoneService == "Yes" else 0],
            "MonthlyCharges": [MonthlyCharges],
            "TotalCharges": [TotalCharges],
            "PaperlessBilling": [1 if PaperlessBilling == "Yes" else 0],
        }
    )

    # Categorical One-Hot Encoded Fields
    cat_values = {
        f"MultipleLines_{MultipleLines}": 1,
        f"InternetService_{InternetService}": 1,
        f"OnlineSecurity_{OnlineSecurity}": 1,
        f"OnlineBackup_{OnlineBackup}": 1,
        f"DeviceProtection_{DeviceProtection}": 1,
        f"TechSupport_{TechSupport}": 1,
        f"StreamingTV_{StreamingTV}": 1,
        f"StreamingMovies_{StreamingMovies}": 1,
        f"Contract_{Contract}": 1,
        f"PaymentMethod_{PaymentMethod}": 1,
    }

    for col in features:
        if col not in df.columns:
            df[col] = cat_values.get(col, 0)

    return df[features]


# ===== Make Prediction =====

if st.button("🔮 Predict Churn"):
    input_df = preprocess_input()

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("❌ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")

    # SHAP Explanation (text summary)
    st.subheader("🔍 SHAP Explanation (Summary)")
    explainer = shap.Explainer(model, input_df)
    shap_values = explainer(input_df)

    # Extract SHAP values for class 1 (churn)
    shap_vals_for_class1 = shap_values.values[0][:, 1]

    df_shap = (
        pd.DataFrame(
            {"Feature": input_df.columns, "SHAP Value": shap_vals_for_class1}
        )
        .sort_values("SHAP Value", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(df_shap)
