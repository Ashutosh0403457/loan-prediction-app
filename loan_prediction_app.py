
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Sidebar for input fields and settings
st.sidebar.title("Applicant Information")

# Input fields in the sidebar with tooltips, defaults, and validations
Gender = st.sidebar.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female", help="Select the gender of the applicant.")
Married = st.sidebar.selectbox("Married", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", help="Is the applicant married?")
Dependents = st.sidebar.number_input(
    "Number of Dependents", 
    min_value=0, 
    max_value=5, 
    step=1, 
    value=0, 
    help="Enter the number of dependents the applicant has."
)
Education = st.sidebar.selectbox("Education", [0, 1], format_func=lambda x: "Graduate" if x == 0 else "Not Graduate", help="Select the educational background of the applicant.")
Self_Employed = st.sidebar.selectbox("Self-Employed", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", help="Is the applicant self-employed?")

ApplicantIncome = st.sidebar.number_input(
    "Applicant Income", 
    min_value=0.0, 
    step=500.0, 
    value=5000.0, 
    help="Enter the applicant's monthly income ."
)
CoapplicantIncome = st.sidebar.number_input(
    "Co-applicant Income", 
    min_value=0.0, 
    step=500.0, 
    value=2000.0, 
    help="Enter the co-applicant's monthly income. Set to 0 if none."
)
LoanAmount = st.sidebar.number_input(
    "Loan Amount (in thousands)", 
    min_value=0.0, 
    step=50.0, 
    value=150.0, 
    help="Enter the desired loan amount in thousands."
)
Loan_Amount_Term = st.sidebar.number_input(
    "Loan Amount Term (days)", 
    min_value=12.0, 
    step=12.0, 
    value=360.0, 
    help="Enter the loan repayment term in days. Default is 360 days."
)
Credit_History = st.sidebar.selectbox("Credit History", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", help="Does the applicant have a good credit history?")

Property_Area = st.sidebar.selectbox("Property Area", [0, 1, 2], format_func=lambda x: ["Rural", "Semiurban", "Urban"][x], help="Select the type of property area.")

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
if st.sidebar.button("Predict Loan Status"):
    # Validate inputs
    if LoanAmount <= 0 or Loan_Amount_Term < 12:
        st.error("Invalid inputs! Please ensure Loan Amount is positive and Loan Amount Term is at least 12 days.")
    else:
        # Scale user input
        scaled_input = scaler.transform(user_input)
        
        # Predict
        prediction = model.predict(scaled_input)
        confidence = model.predict_proba(scaled_input)

        result = "Approved" if prediction[0] == 1 else "Not Approved"
        confidence_score = confidence[0][1] if prediction[0] == 1 else confidence[0][0]

        # Display the result
        st.subheader("Prediction Result:")
        if result == "Approved":
            st.success(f"✅ The loan is **{result}** with a confidence of **{confidence_score:.2f}**.")
        else:
            st.error(f"❌ The loan is **{result}** with a confidence of **{confidence_score:.2f}**.")

        # Confidence chart
        st.write("Confidence Levels:")
        st.bar_chart({"Confidence": [confidence[0][0], confidence[0][1]]}, width=400)
