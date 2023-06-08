import streamlit as st
import numpy as np
from tensorflow import keras

modelNN = keras.models.load_model('predictoNN.h5')
# Create a dictionary to map the input values to the corresponding label encodings
value_mappings = {
    'Sex': {'Female': 0, 'Male': 1},
    'ChestPainType': {'Asymptomatic': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Typical Agina': 3},
    'FastingBS': {'No': 0, 'Yes': 1},
    'RestingECG': {'LVH': 0, 'Normal': 1, 'ST': 2},
    'ExerciseAngina': {'No': 0, 'Yes': 1},
    'ST_Slope': {'Down': 0, 'Flat': 1, 'Up': 2}
}

# Streamlit app code
st.title("Heart Disease Prediction")
st.title('')

# Brief model description
model_description = '''Our Neural Network (NN) model enables doctors to make informed decisions about heart disease 
            by providing fast and accurate predictions. By analyzing important patient information, the model allows doctors to take proactive
            steps in managing assessing patients cardiovascular health. It serves as a valuable tool for early detection empowering patients 
            to make informed lifestyle choices.'''

# Display the brief model description
st.subheader("Model Description")
st.write(model_description)


st.title('')
st.subheader("Prediction form")
st.write('Fill in all the information below and our model will make a prediction')

# Input form
with st.form("heart_disease_form"):
    st.subheader("Patient Information")
    age = st.number_input("Enter the age of the patient in years", min_value=0, max_value=150, value=30)
    st.subheader('')

    sex = st.radio("Select the sex of the patient", ['Male', 'Female'], horizontal=True)
    st.subheader('')

    chest_pain_type = st.radio("Select the type of chest pain experienced by the patient", ['Typical Agina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'], horizontal=True)
    st.subheader('')

    resting_bp = st.number_input("Enter the resting blood pressure of the patient (in mmHg)", min_value=0, max_value=200, value=0)
    st.subheader('')

    cholesterol = st.number_input("Enter the serum cholesterol of the patient (mg/dl)", min_value=0, value=200)
    st.subheader('')

    fasting_bs = st.radio("Is the patient's fasting blood sugar >120mg/dl", ['Yes', 'No'], horizontal=True)
    st.subheader('')

    resting_ecg = st.radio("Select the patient's resting ECG results", ['Normal', 'ST', 'LVH'], horizontal=True, help='ST: ST-T wave abnormality, LVH: left ventricular hypertrophy')
    st.subheader('')

    max_hr = st.number_input("Enter the patient's maximum heart rate", min_value=0, value=60)
    st.subheader('')

    exercise_angina = st.radio("Does the patient experience Exercise-induced angina", ['No', 'Yes'], horizontal=True)
    st.subheader('')

    oldpeak = st.number_input("Enter the patient's oldpeak score", value=1.0, help='ST depression induced by exercise relative to rest.')
    st.subheader('')

    st_slope = st.radio("Select the patient's ST slope pattern", ['Up', 'Flat', 'Down'], horizontal=True, help='The ST segment shift relative to exercise-induced increments in heart rate')
    submit_button = st.form_submit_button(label="Predict")
    
    # Reformat the input data
    input_data = [age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]
    reformatted_data = []
    for feature, value in zip(['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'], input_data):
        if feature in value_mappings:
            encoded_value = value_mappings[feature].get(value)
            reformatted_data.append(encoded_value)
        else:
            reformatted_data.append(value)
            
    # Convert the reformatted data to the desired format
    input_data_formatted = np.array(reformatted_data).reshape(1, -1)
    
    
    
    if submit_button:
        # Make predictions
        predictions = modelNN.predict(input_data_formatted)
        rounded_predictions = np.round(predictions)

        if rounded_predictions == 0:
            prediction_label = 'No heart disease.'
        else:
            prediction_label = 'the Presence of heart disease.'

        st.write('Based on the information provided the prediction indicates:')
        st.write(prediction_label)


# predict no HD
# [53, 'Male', 'Asymptomatic', 124, 260, 'No', 'Normal', 112, 'Yes', 3.0, 'Flat'] 

# predict HD
# [58, 'Male', 'Atypical Angina', 136, 164, 'No', 'ST', 99, 'Yes', 2.0, 'Flat'] 


# to run from terminal > streamlit run app.py