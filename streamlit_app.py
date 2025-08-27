import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier

st.title('🏦 Bank Customer Churn Prediction APP')
st.info('This app predicts whether a customer will churn using Gradient Boosting!')

# ------------------ Load dataset for exploration ------------------
with st.expander(' Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/itsscorps/NPMLChurnPredictionApp/refs/heads/master/cleaned_dataset.csv')
    st.dataframe(df)

    st.write('**X (Features)**')
    X_raw = df.drop('Exited', axis=1)
    st.dataframe(X_raw)

    st.write('**y (Target)**')
    y_raw = df.Exited
    st.dataframe(y_raw)

# ------------------ Data Visualization ------------------
with st.expander('Data Visualization'):
    st.write('**Age vs Balance (colored by Exited)**')
    st.scatter_chart(data=df, x='Balance', y='Age', color='Exited')

    st.write('**Credit Score vs Number of Products (colored by Exited)**')
    st.scatter_chart(data=df, x='NumOfProducts', y='CreditScore', color='Exited')

# ------------------ Sidebar for input ------------------
st.sidebar.header("🔧 Input Features")

CreditScore = st.sidebar.slider("Credit Score", 300, 900, 650)
# Gender
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender_val = 1 if gender == "Male" else 0
Age = st.sidebar.slider("Age", 18, 100, 40)
Tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
Balance = st.sidebar.slider("Balance", 0.0, 250000.0, 50000.0, step=1000.0)
NumOfProducts = st.sidebar.slider("Number of Products", 1, 4, 1)
HasCrCard = st.sidebar.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.sidebar.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.sidebar.slider("Estimated Salary", 0.0, 200000.0, 100000.0, step=1000.0)
# Geography
geography = st.sidebar.selectbox("Select Geography", ["France", "Germany", "Spain"])
geo_dict = {
    "France": {"Geography_Germany": 0, "Geography_Spain": 0},
    "Germany": {"Geography_Germany": 1, "Geography_Spain": 0},
    "Spain": {"Geography_Germany": 0, "Geography_Spain": 1}
}



# Numeric inputs




# Combine into dict
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

# ------------------ Prediction ------------------
input_df = pd.DataFrame([input_data])

# Load trained model
model = joblib.load("gbchurn_model.pkl")

# Predict
prediction = model.predict(input_df)[0]  # 0 = Not churn, 1 = Churn
prediction_proba = model.predict_proba(input_df)[0]  # [prob_not_churn, prob_churn]

# ------------------ Display Probabilities ------------------
st.subheader(" Prediction Probabilities")
df_prediction_proba = pd.DataFrame([{
    "Not Churn": prediction_proba[0],
    "Churn": prediction_proba[1]
}])

st.dataframe(
    df_prediction_proba,
    column_config={
        "Not Churn": st.column_config.ProgressColumn(
            "Not Churn",
            format="%.2f",
            min_value=0.0,
            max_value=1.0,
        ),
        "Churn": st.column_config.ProgressColumn(
            "Churn",
            format="%.2f",
            min_value=0.0,
            max_value=1.0,
        ),
    },
    hide_index=True
)

# ------------------ Final Verdict ------------------
st.subheader(" Final Prediction")
if prediction == 1:
    st.error(f" The customer is LIKELY to CHURN! Probability: {prediction_proba[1]:.2%}")
else:
    st.success(f" The customer is NOT likely to churn. Probability: {prediction_proba[0]:.2%}")

