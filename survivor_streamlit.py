# Import all libraries
from pandas.core.reshape.reshape import get_dummies
import streamlit as st  
import pandas as pd 
import joblib
import os 

# Create the Title
st.title("Survivor Prediction App")

# Create the input widgets
pclass = st.selectbox("what was the passenger's ticket class?",['first','business','economy'])
sex = st.selectbox("What is the passenger's sex?", ['male','female'])
age = st.number_input("Enter the passenger's age")

# Because we applied One Hot Encoding in the ML model, we need to read in the dataset and mirror the steps
# else we will have a mismatch in the number of items input into the model at fit versus the original 
# dimension of the dataset

path = "https://raw.githubusercontent.com/onyekaugochukwu/survivor_prediction/main/survivor_prediction.csv"
dataset = pd.read_csv(path)
dataset_categorical = dataset.drop(['age','survived'],axis=1)
dataset_numerical = dataset.drop(['pclass','sex','survived'],axis=1)

# Use dictionary to map user input th the various features/columns
input_dict = {'pclass':pclass, 'sex':sex, 'age':age}
input_df = pd.DataFrame([input_dict])

# Derive the dataframe of categorical features alone
input_df_without_age = input_df.drop(['age'],axis=1)

# Map user input to one hot encoded dummies
expanded_columns=['pclass_business', 'pclass_economy', 'pclass_first', 'sex_female','sex_male']
new_df = pd.get_dummies(input_df_without_age).reindex(columns=expanded_columns,fill_value=0)
very_new_df = pd.concat([new_df,input_df['age']],axis=1)

# Apply the exapanded dataframe to the pickled file so as to match the dimensions of the .fit and . predict method
pt_model = joblib.load('log_reg')

def predicter():
    m = pt_model.predict(very_new_df)
    return m

predict_button = st.button("Predict Outcome",on_click=predicter)

if predict_button:
    result = predicter()
    st.success(f'The predicted person survival status is {result}')