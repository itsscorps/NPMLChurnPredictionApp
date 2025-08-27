import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier

st.title('Bank Customer Churn Prediction APP')
st.info('This is app builds a machine learning model!')

with st.expander('Data'):
 st.write('**Raw data**')
 df = pd.read_csv('https://raw.githubusercontent.com/itsscorps/NPMLChurnPredictionApp/refs/heads/master/cleaned_dataset.csv')
 df

 st.write('**X**')
 X_raw = df.drop('Exited', axis=1)
 X_raw

 st.write('**y**')
 y_raw = df.Exited
 y_raw

with st.expander('Data visualization'):
  st.write('**Age and Balance**')
  st.scatter_chart(data=df, x='Balance', y='Age', color='Exited')
  st.write('**Credit Score and Has Credit Card or not**')
  st.scatter_chart(data=df, x='NumOfProducts', y='CreditScore', color='Exited')

# Input features
with st.sidebar:
    st.header('Input features')
  # Geography
    geography = st.selectbox("Select Geography", ["France", "Germany", "Spain"])

    geo_dict = {
        "France": {"Geography_Germany": 0, "Geography_Spain": 0},
        "Germany": {"Geography_Germany": 1, "Geography_Spain": 0},
        "Spain": {"Geography_Germany": 0, "Geography_Spain": 1}
    }

    # Gender
    gender = st.selectbox("Gender", ["Female", "Male"])
    gender_val = 1 if gender == "Male" else 0

    # Numeric inputs
    CreditScore = st.slider("Credit Score", min_value=300, max_value=900, value=650, step=1)
    Age = st.slider("Age", min_value=18, max_value=100, value=40, step=1)
    Tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=5, step=1)
    Balance = st.slider("Balance", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)
    NumOfProducts = st.slider("Number of Products", min_value=1, max_value=4, value=1, step=1)
    HasCrCard = st.selectbox("Has Credit Card", [0, 1])
    IsActiveMember = st.selectbox("Is Active Member", [0, 1])
    EstimatedSalary = st.slider("Estimated Salary", min_value=0.0, max_value=200000.0, value=100000.0, step=1000.0)

    # Combine into input dict
    input_data = {
        "CreditScore": CreditScore,
        "Gender": gender_val,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
        **geo_dict[geography]
    }

# Assuming you already have `input_data` from the sidebar
input_df = pd.DataFrame([input_data])

# Load the trained model (if not already loaded)
model = joblib.load("gbchurn_model.pkl")

# Make prediction
prediction = model.predict(input_df)[0]  # 0 = Not churn, 1 = Churn
prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of churn

# Display in Streamlit
st.subheader("Prediction")
if prediction == 1:
    st.write(f"The customer is likely to churn. Probability: {prediction_proba:.2%}")
else:
    st.write(f"The customer is NOT likely to churn. Probability: {prediction_proba:.2%}")
