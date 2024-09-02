import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load the trained model
import traceback


# Load the model (replace 'your_model.pkl' with your actual model file)
try:
    model = joblib.load('my_model.joblib')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.text("Traceback:")
    st.text(traceback.format_exc())
    st.stop()

# Label mapping for prediction output
label_mapping = {
    0: 'Normal_Weight',
    1: 'Overweight_Level_I', 
    2: 'Overweight_Level_II',
    3: 'Obesity_Type_I', 
    4: 'Insufficient_Weight', 
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}

# Streamlit app title
st.title("Obesity Level Prediction")

# Input fields for the numerical features
st.header("Please input the following details:")

age = st.number_input("Age", min_value=10, max_value=100, step=1)
height = st.number_input("Height (cm)", min_value=100, max_value=250, step=1)
weight = st.number_input("Weight (kg)", min_value=20, max_value=200, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
fcvc = st.number_input("Frequency of consumption of vegetables (FCVC)", min_value=0, max_value=3, step=1)
ncp = st.number_input("Number of main meals (NCP)", min_value=1, max_value=4, step=1)
ch2o = st.number_input("Consumption of water daily (CH2O)", min_value=1, max_value=3, step=1)
faf = st.number_input("Physical activity frequency (FAF)", min_value=0, max_value=3, step=1)
tue = st.number_input("Time of using technology devices (TUE)", min_value=0, max_value=3, step=1)

# Ordinal features with select boxes for input
calc = st.selectbox("Consumption of alcohol (CALC)", ['Never', 'Sometimes', 'Frequently', 'Always'])
caec = st.selectbox("Consumption of food between meals (CAEC)", ['No', 'Sometimes', 'Frequently', 'Always'])

# One-hot encoded categorical features
gender = st.selectbox("Gender", ['Female', 'Male'])
family_history = st.selectbox("Family history with overweight", ['No', 'Yes'])
favc = st.selectbox("Frequent consumption of high-caloric food (FAVC)", ['No', 'Yes'])
smoke = st.selectbox("Smoking habit", ['No', 'Yes'])
scc = st.selectbox("Monitoring of calorie consumption (SCC)", ['No', 'Yes'])

# Transportation method with multiple options (you can modify this depending on your specific requirement)
mtrans = st.selectbox("Transportation", ['Walking', 'Public Transportation', 'Automobile', 'Motorbike', 'Bike'])

# Create the input dataframe
input_data = pd.DataFrame({
    'numerical__Age': [age],
    'numerical__Height': [height],
    'numerical__Weight': [weight],
    'numerical__BMI': [bmi],
    'numerical__FCVC': [fcvc],
    'numerical__NCP': [ncp],
    'numerical__CH2O': [ch2o],
    'numerical__FAF': [faf],
    'numerical__TUE': [tue],
    'ordinal__CALC': [calc],
    'ordinal__CAEC': [caec],
    'onehot__Gender_Female': [1 if gender == 'Female' else 0],
    'onehot__Gender_Male': [1 if gender == 'Male' else 0],
    'onehot__family_history_with_overweight_no': [1 if family_history == 'No' else 0],
    'onehot__family_history_with_overweight_yes': [1 if family_history == 'Yes' else 0],
    'onehot__FAVC_no': [1 if favc == 'No' else 0],
    'onehot__FAVC_yes': [1 if favc == 'Yes' else 0],
    'onehot__SMOKE_no': [1 if smoke == 'No' else 0],
    'onehot__SMOKE_yes': [1 if smoke == 'Yes' else 0],
    'onehot__SCC_no': [1 if scc == 'No' else 0],
    'onehot__SCC_yes': [1 if scc == 'Yes' else 0],
    'onehot__MTRANS_Walking': [1 if mtrans == 'Walking' else 0],
    'onehot__MTRANS_Public_Transportation': [1 if mtrans == 'Public Transportation' else 0],
    'onehot__MTRANS_Automobile': [1 if mtrans == 'Automobile' else 0],
    'onehot__MTRANS_Motorbike': [1 if mtrans == 'Motorbike' else 0],
    'onehot__MTRANS_Bike': [1 if mtrans == 'Bike' else 0]
})

# Button to predict
if st.button("Predict Obesity Level"):
    prediction = model.predict(input_data)  # Assuming the model has a predict method
    st.subheader(f"The predicted obesity level is: {label_mapping[prediction[0]]}")

# Optional: Display the input dataframe for verification
st.write("Input Data for Prediction:")
st.dataframe(input_data)
