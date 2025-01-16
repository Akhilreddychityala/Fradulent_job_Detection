import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('datasets/fake_job_postings.csv')

# Check the columns in the dataset
print(df.columns)

# Select features and target variable
X = df[['title', 'location', 'salary_range', 'company_profile', 'description',
         'requirements', 'benefits', 'employment_type', 'required_experience']]  # Updated column names
y = df['fraudulent']  # Assuming 'fraudulent' is your target column

# Handle missing values if any
X.fillna('', inplace=True)

# Encoding categorical features
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('pickle/app.model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Optionally, save label encoders if needed for future predictions
with open('pickle/label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

print("Model training complete and saved.")
