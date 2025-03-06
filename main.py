import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load or train the model
MODEL_FILE = 'credit_risk_model.pkl'
try:
    model = joblib.load(MODEL_FILE)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# Sample data
data = {
    'Age': [25, 45, 35, 50, 23],
    'Income': [40000, 80000, 60000, 100000, 30000],
    'Credit Score': [700, 650, 600, 750, 580],
    'Loan Amount': [10000, 20000, 15000, 30000, 12000],
    'Default': [0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# Dashboard layout
st.title("Credit Risk Dashboard")

# Data overview
st.header("Data Overview")
st.dataframe(df)

# Visualizations
st.header("Data Visualizations")

fig_income = px.histogram(df, x='Income', color='Default', title="Income Distribution by Default Status")
st.plotly_chart(fig_income)

fig_credit_score = px.scatter(df, x='Credit Score', y='Loan Amount', color='Default',
                              title="Credit Score vs Loan Amount")
st.plotly_chart(fig_credit_score)

# Model training or loading
st.header("Credit Risk Prediction")

X = df[['Age', 'Income', 'Credit Score', 'Loan Amount']]
y = df['Default']

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
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=0, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
loan_amount = st.number_input("Loan Amount", min_value=0, value=15000)

if st.button("Predict Risk"):
    input_data = pd.DataFrame([[age, income, credit_score, loan_amount]], 
                              columns=['Age', 'Income', 'Credit Score', 'Loan Amount'])
    prediction = model.predict(input_data)
    risk_status = "High Risk" if prediction[0] == 1 else "Low Risk"
    st.subheader(f"Prediction: {risk_status}")
