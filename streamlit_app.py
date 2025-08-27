import streamlit as st
import pandas as pd
import joblib

st.title('Bank Customer Churn Prediction APP')
st.info('This app predicts whether a customer will churn using Gradient Boosting!')

# Load dataset for exploration
with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/itsscorps/NPMLChurnPredictionApp/refs/heads/master/cleaned_dataset.csv')
    st.dataframe(df)

    st.write('**X (Features)**')
    X_raw = df.drop('Exited', axis=1)
    st.dataframe(X_raw)

    st.write('**y (Target)**')
    y_raw = df.Exited
    st.dataframe(y_raw)

# Data Visualization
with st.expander('Data visualization'):
    st.write('**Age vs Balance (colored by Exited)**')
    st.scatter_chart(data=df, x='Balance', y='Age', color='Exited')
    st.write('**Credit Score vs Number of Products (colored by Exited)**')
    st.scatter_chart(data=df, x='NumOfProducts', y='CreditScore', color='Exited')

# ---------------- Sidebar inputs ----------------
with st.sidebar:
    st.header("Input Features")

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
    CreditScore = st.slider("Credit Score", 300, 900, 650)
    Age = st.slider("Age", 18, 100, 40)
    Tenure = st.slider("Tenure (years)", 0, 10, 5)
    Balance = st.slider("Balance", 0.0, 250000.0, 50000.0, step=1000.0)
    NumOfProducts = st.slider("Number of Products", 1, 4, 1)
    HasCrCard = st.selectbox("Has Credit Card", [0, 1])
    IsActiveMember = st.selectbox("Is Active Member", [0, 1])
    EstimatedSalary = st.slider("Estimated Salary", 0.0, 200000.0, 100000.0, step=1000.0)
