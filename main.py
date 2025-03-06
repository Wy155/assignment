import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load or train the model
MODEL_FILE = 'credit_risk.joblib'
try:
    model = joblib.load(MODEL_FILE)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# Sample data
data = {
    'cb_person_cred_hist_length': [2, 5, 3, 6, 4],
    'cb_person_default_on_file_Y': [0, 1, 0, 0, 1],
    'loan_amnt': [10000, 20000, 15000, 30000, 12000],
    'loan_int_rate': [10.5, 15.0, 12.0, 9.5, 16.0],
    'loan_intent_EDUCATION': [1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# Dashboard layout
st.title("Credit Risk Dashboard")

# Data overview
st.header("Data Overview")
st.dataframe(df)

# Visualizations
st.header("Data Visualizations")

fig_interest_rate = px.histogram(df, x='loan_int_rate', color='cb_person_default_on_file_Y', title="Loan Interest Rate by Default Status")
st.plotly_chart(fig_interest_rate)

fig_loan_amount = px.scatter(df, x='loan_amnt', y='cb_person_cred_hist_length', color='cb_person_default_on_file_Y',
                              title="Loan Amount vs Credit History Length")
st.plotly_chart(fig_loan_amount)

# Model training or loading
st.header("Credit Risk Prediction")

X = df[['cb_person_cred_hist_length', 'cb_person_default_on_file_Y', 'loan_amnt', 'loan_int_rate', 'loan_intent_EDUCATION']]
y = df['cb_person_default_on_file_Y']

if not model_loaded:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    st.success(f"Model trained and saved as {MODEL_FILE}")
    y_pred = model.predict(X_test)
    st.subheader("Model Evaluation")
    st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
else:
    st.success(f"Model loaded from {MODEL_FILE}")

# User input for risk assessment
st.header("Assess Credit Risk")
credit_hist_length = st.number_input("Credit History Length", min_value=0, value=3)
default_on_file = st.selectbox("Default on File (Yes: 1, No: 0)", options=[0, 1])
loan_amount = st.number_input("Loan Amount", min_value=0, value=15000)
loan_interest_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=100.0, value=12.0)
loan_intent_education = st.selectbox("Loan Intent - Education (Yes: 1, No: 0)", options=[0, 1])

if st.button("Predict Risk"):
    input_data = pd.DataFrame([[credit_hist_length, default_on_file, loan_amount, loan_interest_rate, loan_intent_education]], 
                              columns=['cb_person_cred_hist_length', 'cb_person_default_on_file_Y', 'loan_amnt', 'loan_int_rate', 'loan_intent_EDUCATION'])
    prediction = model.predict(input_data)
    risk_status = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.subheader(f"Prediction: {risk_status}")
