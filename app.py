import streamlit as st
import pandas as pd
import joblib
# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Title
st.title("📊 Telecom Customer Churn Prediction")
st.markdown("Enter the customer details below to predict churn.")

# Input fields (on main screen)
customer_name = st.text_input("Customer Name")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.selectbox("Tenure (in months)", list(range(0, 73)))

with col2:
    monthly_charge = st.selectbox("Monthly Charges", list(range(0, 151)))

with col3:
    total_charge = st.selectbox("Total Charges", list(range(0, 8001, 50)))

gender = st.radio("Gender", ["Male", "Female"])
is_male = 1 if gender == "Male" else 0

new_customer = st.radio("Is this a new customer?", ["Yes", "No"])
# Not used in prediction directly unless you've added it in training

payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

# Payment method one-hot encoding
payment_electronic = int(payment_method == "Electronic check")
payment_mailed = int(payment_method == "Mailed check")
payment_bank = int(payment_method == "Bank transfer (automatic)")
payment_card = int(payment_method == "Credit card (automatic)")

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
contract_two_year = int(contract == "Two year")

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
fiber_optic = int(internet_service == "Fiber optic")

# Prediction button
if st.button("Predict Churn"):
    input_data = pd.DataFrame([[
        tenure, monthly_charge, total_charge, is_male,
        payment_electronic, payment_mailed, payment_bank, payment_card,
        contract_two_year, fiber_optic
    ]], columns=[
        'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'Contract_Two year', 'InternetService_Fiber optic'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ {customer_name} is likely to churn. (Probability: {probability:.2%})")
    else:
        st.success(f"✅ {customer_name} is not likely to churn. (Probability: {probability:.2%})")
