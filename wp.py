import streamlit as st
import pickle
import pandas as pd

# Load the trained model using pickle
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    model_path = r"C:\Users\Lenovo\Desktop\Cloud\build.pkl"  # Replace with your .pkl file path
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title of the web app
st.title("Car Insurance Predictor")

# Create a form for user input
with st.form("car_insurance_form"):
    # Input fields
    age = st.number_input("Customer Age", min_value=0, max_value=100, value=30, step=1)
    balance = st.number_input("Bank Balance", min_value=0, value=5000, step=100)
    hh_insurance = st.selectbox("Health Insurance (0 for No)", options=[0, 1], index=0)
    
    # CarLoan dropdown
    car_loan = st.selectbox("CarLoan", options=["NO", "Yes"], index=0)
    car_loan = 1 if car_loan == "Yes" else 0  # Convert to binary (0 or 1)
    
    # Job dropdown
    job = st.selectbox(
        "Job",
        options=[
            "Admin", "Blue-collar", "Entrepreneur", "Housemaid", "Management",
            "Retired", "Self-employed", "Services", "Student", "Technician", "Unemployed"
        ],
        index=0
    )
    # Map job to numerical value (ensure this matches your model's encoding)
    job_mapping = {
        "Admin": 0, "Blue-collar": 1, "Entrepreneur": 2, "Housemaid": 3,
        "Management": 4, "Retired": 5, "Self-employed": 6, "Services": 7,
        "Student": 8, "Technician": 9, "Unemployed": 10
    }
    job = job_mapping[job]
    
    # Marital Status dropdown
    marital_status = st.selectbox(
        "Marital Status",
        options=["Divorced", "Married", "Single"],
        index=0
    )
    # Map marital status to numerical value
    marital_mapping = {"Divorced": 0, "Married": 1, "Single": 2}
    marital_status = marital_mapping[marital_status]
    
    # Education Level dropdown
    education_level = st.selectbox(
        "Education Level",
        options=["Primary", "Secondary", "Tertiary"],
        index=0
    )
    # Map education level to numerical value
    education_mapping = {"Primary": 0, "Secondary": 1, "Tertiary": 2}
    education_level = education_mapping[education_level]
    
    # Month dropdown
    month = st.selectbox(
        "Month",
        options=[
            "Apr", "Aug", "Dec", "Feb", "Jan", "Jul", "Jun", "Mar", "May", "Nov", "Oct", "Sep"
        ],
        index=0
    )
    # Map month to numerical value
    month_mapping = {
        "Apr": 0, "Aug": 1, "Dec": 2, "Feb": 3, "Jan": 4, "Jul": 5,
        "Jun": 6, "Mar": 7, "May": 8, "Nov": 9, "Oct": 10, "Sep": 11
    }
    month = month_mapping[month]
    
    # Submit button
    submitted = st.form_submit_button("Predict")

# When the form is submitted
if submitted:
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        "Age": [age],
        "Balance": [balance],
        "HHInsurance": [hh_insurance],
        "CarLoan": [car_loan],
        "Job": [job],
        "Marital": [marital_status],  # Match the feature name used during training
        "Education": [education_level],  # Match the feature name used during training
        "LastContactMonth": [month],  # Match the feature name used during training
    })
    
    # Reorder columns to match the order used during training
    # Replace this with the correct column order from your training data
    correct_feature_order = [
        "Age", "Balance", "HHInsurance", "CarLoan", "Job", "Marital", 
        "Education", "LastContactMonth"
    ]
    input_data = input_data[correct_feature_order]
    
    # Debug: Print input data
    st.write("Input Data:", input_data)
    
    # Make a prediction using the model
    try:
        # Check if the model has predict_proba (for probabilities)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)
            st.write("Predicted Probabilities:", probabilities)
            predictions = (probabilities[:, 1] > 0.5).astype(int)  # Use a threshold of 0.5
        else:
            predictions = model.predict(input_data)
        
        st.success(f"Prediction: {predictions[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")