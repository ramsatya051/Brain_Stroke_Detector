"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


@st.cache_data()
def load_data():
    """This function returns the preprocessed data"""

    # Load the dataset into DataFrame.
    df = pd.read_csv('healthcare_data_set_new.csv')
    df.dropna(inplace=True)
    df['gender'] = LabelEncoder().fit_transform(df['gender'])
    df['work_type'] = LabelEncoder().fit_transform(df['work_type'])
    df['ever_married'] = LabelEncoder().fit_transform(df['ever_married'])
    df['Residence_type'] = LabelEncoder().fit_transform(df['Residence_type'])
    df['smoking_status'] = LabelEncoder().fit_transform(df['smoking_status'])

    # Perform feature and target split
    X = df[["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]]
    y = df["stroke"]

    return df, X, y

@st.cache_resource()
def train_model(X, y):
    """This function trains the model and return the model and model score"""
    # Create the model
    '''model = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='entropy',
            max_depth=4, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1, 
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=42, splitter='best'
        )'''
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    # Fit the data on model
    model.fit(x_train, y_train)
    # Get the model score
    score = model.score(x_test, y_test)

    # Return the values
    return model, score

def predict(X, y, features):
    # Get model and model score
    model, score = train_model(X, y)
    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score
