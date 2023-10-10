import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("/home/jovyan/day_2/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Create a Streamlit app
st.title("Loan Approval Predictor")


# Input fields for user input
int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)
annual_inc = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0)
delinq_2yrs = st.number_input("Number of Delinquencies in the Last 2 Years", min_value=0, value=0)
fico_range_high = st.number_input("FICO Score (High Range)", min_value=300, max_value=850, value=700)
revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, value=10000.0)
open_acc = st.number_input("Number of Open Credit Lines", min_value=0, value=5)

# Make a prediction when the user clicks the "Predict" button
if st.button("Predict Loan Approval"):
    # Create a feature array from user inputs
    features = np.array([int_rate, emp_length, annual_inc, delinq_2yrs, fico_range_high, revol_bal, open_acc]).reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(features)
    
    # Display the prediction result
    if prediction[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Rejected.")
