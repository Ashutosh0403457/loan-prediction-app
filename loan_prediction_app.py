%%writefile loanappr.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier 

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and preprocess dataset
df = pd.read_csv("LoanApprovalPred_Adjusted_v2.csv")

# Data preprocessing
df.drop(columns=['Unnamed: 13', '96', 'Loan_ID'], errors='ignore', inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

le = LabelEncoder()
df[df.select_dtypes('object').columns] = df.select_dtypes('object').apply(le.fit_transform)

# Feature-target split and scaling
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Loan Approval Prediction System")
st.write("Enter applicant details below to predict the loan approval status.")

# Input fields for applicant details
col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    Married = st.selectbox("Married", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    Dependents = st.number_input("Number of Dependents", min_value=0, max_value=5, step=1)
    Education = st.selectbox("Education", [0, 1], format_func=lambda x: "Graduate" if x == 0 else "Not Graduate")
    Self_Employed = st.selectbox("Self-Employed", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    ApplicantIncome = st.number_input("Applicant Income ", min_value=0.0, step=500.0)
    CoapplicantIncome = st.number_input("Co-applicant Income ", min_value=0.0, step=500.0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0.0, step=50.0)
    Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0.0, step=12.0)
    Credit_History = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

Property_Area = st.selectbox("Property Area", [0, 1, 2], format_func=lambda x: ["Rural", "Semiurban", "Urban"][x])

# Collect input data
user_input = pd.DataFrame([{
    'Gender': Gender,
    'Married': Married,
    'Dependents': Dependents,
    'Education': Education,
    'Self_Employed': Self_Employed,
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'Property_Area': Property_Area
}])

# Prediction button
if st.button("Predict Loan Status"):
    scaled_input = scaler.transform(user_input)  # Scale user input
    prediction = model.predict(scaled_input)  # Predict
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    st.subheader("Prediction Result:")
    st.success(f"The loan is {result}.")


