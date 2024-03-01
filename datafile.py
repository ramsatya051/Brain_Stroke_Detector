import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("brain_stroke_dataset_more_accurate.csv")

print(df['gender'].unique())
df['gender'] = LabelEncoder().fit_transform(df['gender'])
print(df['gender'].unique())

print(df['work_type'].unique())
df['work_type'] = LabelEncoder().fit_transform(df['work_type'])
print(df['work_type'].unique())

print(df['ever_married'].unique())
df['ever_married'] = LabelEncoder().fit_transform(df['ever_married'])
print(df['ever_married'].unique())

print(df['residence_type'].unique())
df['residence_type'] = LabelEncoder().fit_transform(df['residence_type'])
print(df['residence_type'].unique())

print(df.head(100))

# Set random seed for reproducibility
'''random.seed(42)

# Define the attributes and their possible values
gender = ['Male', 'Female']
hypertension = [0, 1]
heart_disease = [0, 1]
ever_married = ['Yes', 'No']
work_type = ['Private', 'Self-employed', 'Govt_job', 'Never_worked']
residence_type = ['Urban', 'Rural']
stroke = [0, 1]

# Generate data for the dataset
data = []

# Generate more data points to achieve around 2000 entries
for _ in range(2000):
    row = [
        random.choice(gender),
        np.random.normal(55, 15),  # Normal distribution around mean age of 55 with std deviation 15
        random.choices(hypertension, weights=[0.85, 0.15])[0],  # Higher probability for no hypertension
        random.choices(heart_disease, weights=[0.92, 0.08])[0],  # Higher probability for no heart disease
        random.choices(ever_married, weights=[0.75, 0.25])[0],  # Higher probability for being married
        random.choices(work_type, weights=[0.6, 0.25, 0.1, 0.05])[0],  # Custom weights based on real-world data
        random.choice(residence_type),
        np.random.normal(100, 40),  # Normal distribution around mean glucose level of 100 with std deviation 40
        np.random.normal(28, 7),    # Normal distribution around mean BMI of 28 with std deviation 7
        random.choices(stroke, weights=[0.95, 0.05])[0]  # Higher probability for no stroke
    ]
    data.append(row)

# Create a DataFrame
df = pd.DataFrame(data, columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                                 'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'stroke'])

# Round age, avg_glucose_level, and bmi to integers
df['age'] = df['age'].astype(int)
df['avg_glucose_level'] = df['avg_glucose_level'].astype(int)
df['bmi'] = df['bmi'].astype(int)

# Save the DataFrame to a CSV file
df.to_csv('brain_stroke_dataset_more_accurate.csv', index=False)

print("Dataset saved to 'brain_stroke_dataset_more_accurate.csv'")'''
