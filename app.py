import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
with open(r'C:\Users\akhil\job\Real_Or_Fake_Job_Prediction-master\pickle\app.model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the label encoders
with open(r'C:\Users\akhil\job\Real_Or_Fake_Job_Prediction-master\pickle\label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Title of the app
st.title('Job Posting Prediction')

# Input fields
job_id = st.text_input('Job ID (optional)')  # This field is optional
job_title = st.text_input('Job Title')
location = st.text_input('Location')
salary_range = st.text_input('Salary Range')
company_profile = st.text_area('Company Profile')
job_description = st.text_area('Job Description')
job_requirements = st.text_area('Job Requirements')
benefits = st.text_area('Benefits')
employment_type = st.selectbox('Employment Type', options=['Full-time', 'Part-time', 'Contract'])
required_experience = st.text_input('Required Experience')

# Prediction button
if st.button('Press Enter to apply'):
    # Create the input DataFrame
    input_data = pd.DataFrame({
        'title': [job_title],
        'location': [location],
        'salary_range': [salary_range],
        'company_profile': [company_profile],
        'description': [job_description],
        'requirements': [job_requirements],
        'benefits': [benefits],
        'employment_type': [employment_type],
        'required_experience': [required_experience]
    })

    # Handle unseen labels in label encoding
    for column in input_data.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            le = label_encoders[column]
            try:
                input_data[column] = le.transform(input_data[column])
            except ValueError:
                # Assign NaN or a default value for unseen labels
                input_data[column] = np.nan

    try:
        # Make prediction
        prediction = model.predict(input_data)
        st.write(f'The job posting is predicted to be: {"Fake" if prediction[0] == 0 else "Real"}')
    except Exception as e:
        st.error(f'An error occurred during prediction: {str(e)}')
