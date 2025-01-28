import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv('LoanApprovalPred.csv')
    df.drop(columns=['Unnamed: 13', '96', 'Loan_ID'], errors='ignore', inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    le = LabelEncoder()
    df[df.select_dtypes('object').columns] = df.select_dtypes('object').apply(le.fit_transform)
    return df

df = load_data()

# Feature-target split and scaling
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']
X_scaled = StandardScaler().fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Streamlit App
st.title("Loan Approval Prediction")

# Display model performance
st.write("### Model Performance")
st.write("Training Accuracy: {:.2f}%".format(100 * metrics.accuracy_score(y_train, model.predict(X_train))))
st.write("Testing Accuracy: {:.2f}%".format(100 * metrics.accuracy_score(y_test, model.predict(X_test))))

# Dynamic Prediction Input
st.write("### Enter Applicant Details for Prediction")
person_details = {
    'Gender': st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male"),
    'Married': st.selectbox("Married", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    'Dependents': st.number_input("Number of Dependents", min_value=0, step=1),
    'Education': st.selectbox("Education", [0, 1], format_func=lambda x: "Graduate" if x == 0 else "Not Graduate"),
    'Self_Employed': st.selectbox("Self-Employed", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    'ApplicantIncome': st.number_input("Applicant's Income", min_value=0.0, step=1000.0),
    'CoapplicantIncome': st.number_input("Co-applicant's Income", min_value=0.0, step=1000.0),
    'LoanAmount': st.number_input("Loan Amount (in thousands)", min_value=0.0, step=1.0),
    'Loan_Amount_Term': st.number_input("Loan Amount Term (in days)", min_value=0.0, step=1.0),
    'Credit_History': st.selectbox("Credit History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    'Property_Area': st.selectbox("Property Area", [0, 1, 2], format_func=lambda x: ["Rural", "Semiurban", "Urban"][x])
}

if st.button("Predict Loan Approval"):
    input_df = pd.DataFrame([person_details])
    loan_approval = model.predict(StandardScaler().fit(X).transform(input_df))
    result = "Approved" if loan_approval[0] == 1 else "Not Approved"
    st.write(f"### Loan Approval Prediction: **{result}**")
