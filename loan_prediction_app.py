import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load and preprocess dataset
df = pd.read_csv('LoanApprovalPred.csv')
df.drop(columns=['Unnamed: 13', '96', 'Loan_ID'], errors='ignore', inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

le = LabelEncoder()
df[df.select_dtypes('object').columns] = df.select_dtypes('object').apply(le.fit_transform)

# Feature-target split and scaling
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']
X_scaled = StandardScaler().fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header("Enter Your Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.number_input("Number of Dependents", min_value=0, max_value=10)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income = st.sidebar.number_input("Applicant's Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant's Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_amount_term = st.sidebar.number_input("Loan Amount Term (in months)", min_value=0)
credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Prediction button
if st.sidebar.button("Predict"):
    person_details = pd.DataFrame([{
        'Gender': 1 if gender == 'Male' else 0,
        'Married': 1 if married == 'Yes' else 0,
        'Dependents': dependents,
        'Education': 1 if education == 'Not Graduate' else 0,
        'Self_Employed': 1 if self_employed == 'Yes' else 0,
        'ApplicantIncome': income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': 1 if credit_history == 'Yes' else 0,
        'Property_Area': ['Rural', 'Semiurban', 'Urban'].index(property_area)
    }])

    # Scale the input and make prediction
    loan_approval = model.predict(StandardScaler().fit(X).transform(person_details))
    
    if loan_approval[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Not Approved.")

# Visualizing Loan Approval Distribution
st.write("### Loan Approval Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Loan_Status', data=df, ax=ax)
ax.set_title("Loan Approval Distribution")
st.pyplot(fig)
