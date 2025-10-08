import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt
from sklearn.ensemble import GradientBoostingClassifier


# APP TITLE AND DESCRIPTION

st.title('Bank Customer Churn Prediction App')
st.info('This app predicts whether a customer will churn using a trained Gradient Boosting model.')


# LOAD AND EXPLORE DATA

with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv(
        'https://raw.githubusercontent.com/itsscorps/NPMLChurnPredictionApp/refs/heads/master/cleaned_dataset.csv'
    )
    st.dataframe(df)

    # Show features (X)
    st.write('**X (Features)**')
    X_raw = df.drop('Exited', axis=1)
    st.dataframe(X_raw)

    # Show target (y)
    st.write('**y (Target)**')
    y_raw = df.Exited
    st.dataframe(y_raw)


# DATA VISUALIZATION

with st.expander('Data Visualization'):
      
    chart = alt.Chart(df).mark_circle(size=60).encode(
    x="Age",
    y="Balance",
    color=alt.Color("Exited:N",
                    scale=alt.Scale(domain=[0, 1], range=["#1f77b4", "#d62728"]),  # blue for 0, red for 1
                    legend=alt.Legend(title="Churn (Exited)"))
    ).properties(
    title="Balance vs Age Colored by Churn"
    )

    st.altair_chart(chart, use_container_width=True)
    
    st.write('### Balance Distribution')
    st.write('Average Balance by Exited')
    balance_mean = df.groupby('Exited')['Balance'].mean()
    st.line_chart(balance_mean)

    st.write('### Age Distribution')
    st.write('Average Age by Exited')
    age_mean = df.groupby('Exited')['Age'].mean()
    st.line_chart(age_mean)

    chart = alt.Chart(df).mark_circle(size=60).encode(
    x="CreditScore",
    y="NumOfProducts",
    color=alt.Color("Exited:N",
                    scale=alt.Scale(domain=[0, 1], range=["#1f77b4", "#d62728"]),  # blue for 0, red for 1
                    legend=alt.Legend(title="Churn (Exited)"))
    ).properties(
    title="CreditScore vs Number of Products Colored by Churn"
    )

    st.altair_chart(chart, use_container_width=True)

    st.write('### Credit Score Distribution')
    st.write('Average Credit Score by Exited')
    credit_mean = df.groupby('Exited')['CreditScore'].mean()
    st.line_chart(credit_mean)

    st.write('### Number of Products  Distribution')
    st.write('Number of customers per NumOfProducts grouped by Exited')
    prod_counts = df.groupby(['NumOfProducts', 'Exited']).size().unstack(fill_value=0)
    st.line_chart(prod_counts)


# SIDEBAR FOR USER INPUT

st.sidebar.header("Input Features")

# Geography (One-Hot Encoded: France = baseline)
geography = st.sidebar.selectbox("Select Geography", ["France", "Germany", "Spain"])
geo_dict = {
    "France": {"Geography_Germany": 0, "Geography_Spain": 0},
    "Germany": {"Geography_Germany": 1, "Geography_Spain": 0},
    "Spain": {"Geography_Germany": 0, "Geography_Spain": 1}
}

# Gender (Binary Encoding: Female = 0, Male = 1)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender_val = 1 if gender == "Male" else 0

# Numerical input sliders
CreditScore = st.sidebar.slider("Credit Score", 300, 900, 650)
Age = st.sidebar.slider("Age", 18, 100, 40)
Tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
Balance = st.sidebar.slider("Balance", 0.0, 250000.0, 50000.0, step=1000.0)
NumOfProducts = st.sidebar.slider("Number of Products", 1, 4, 1)
HasCrCard = st.sidebar.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.sidebar.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.sidebar.slider("Estimated Salary", 0.0, 200000.0, 100000.0, step=1000.0)

# Combine all inputs into a dictionary
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
    **geo_dict[geography]  # Add one-hot encoded geography
}

# Convert to DataFrame (needed for sklearn)
input_df = pd.DataFrame([input_data])


# ALIGN COLUMNS WITH TRAINING DATA

# Ensure input_df has same feature columns as training dataset
df_train = pd.read_csv(
    'https://raw.githubusercontent.com/itsscorps/NPMLChurnPredictionApp/refs/heads/master/cleaned_dataset.csv'
)
feature_columns = df_train.drop("Exited", axis=1).columns
input_df = input_df.reindex(columns=feature_columns, fill_value=0)


# LOAD TRAINED MODEL
model = joblib.load("gbchurn_model.pkl")


# MAKE PREDICTION
prediction = model.predict(input_df)[0]  # Predicted class: 0 = Not churn, 1 = Churn
prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of churn (class = 1)


# DISPLAY RESULTS
st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"Customer will churn.\n\nProbability of churn: {prediction_proba:.2%}")
else:
    st.success(f"Customer will not churn.\n\nProbability of churn: {prediction_proba:.2%}")



# DISPLAY PROBABILITY TABLE

# Create probability dataframe
df_prediction_proba = pd.DataFrame([{
    "Not Churn": 1 - prediction_proba,
    "Churn": prediction_proba
}])

# Show probabilities with progress bars
st.subheader("Prediction Probabilities")
st.dataframe(
    df_prediction_proba,
    column_config={
        "Not Churn": st.column_config.ProgressColumn(
            "Not Churn",
            format="%.2f",  # Show decimals (can be changed to percentage if needed)
            min_value=0,
            max_value=1,
        ),
        "Churn": st.column_config.ProgressColumn(
            "Churn",
            format="%.2f",
            min_value=0,
            max_value=1,
        ),
    },
    hide_index=True
)
# Add a navigation button at the bottom
st.page_link("2nd Page.py", label="➡ Go to Loan Default Prediction Page", icon="🔮")
