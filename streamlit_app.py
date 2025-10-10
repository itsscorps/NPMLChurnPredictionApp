import streamlit as st
st.markdown(
    """
    <style>
    /* === GLOBAL BACKGROUND IMAGE === */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                    url("https://images.unsplash.com/photo-1521791136064-7986c2920216");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }

    /* Remove Streamlit's default solid background that hides image */
    .block-container {
        background: transparent !important;
    }

    /* === MAIN BODY === */
    .main .block-container {
        background-color: rgba(0, 0, 0, 0.55) !important;  /* darker overlay */
        border-radius: 12px;
        padding: 2rem;
        color: white !important;
    }

    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.55) !important;
        backdrop-filter: blur(4px);
        color: white !important;
    }

    section[data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 500;
    }

    /* Sidebar widgets */
    section[data-testid="stSidebar"] .stSelectbox div,
    section[data-testid="stSidebar"] .stMultiSelect div,
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stFileUploader div,
    section[data-testid="stSidebar"] .stRadio div,
    section[data-testid="stSidebar"] .stCheckbox div {
        background-color: rgba(0, 0, 0, 0.35) !important;
        color: white !important;
        border-radius: 6px;
        padding: 4px;
    }

    /* Dropdown options */
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        background-color: rgba(0, 0, 0, 0.85) !important;
        color: white !important;
    }

    /* Slider labels */
    section[data-testid="stSidebar"] .stSlider span {
        color: white !important;
    }

    /* === FILE UPLOAD BUTTON === */
    [data-testid="stFileUploader"] section div div {
        background-color: rgba(0, 0, 0, 0.6) !important;
        color: white !important;
        border: 1px solid white;
        border-radius: 8px;
    }
    [data-testid="stFileUploader"] section div div span {
        color: #f9f9f9 !important;
    }

    /* === GENERAL BODY TEXT === */
    .stMarkdown, .stText, .stCaption, .stExpander {
        color: white !important;
    }
    .stCaption {
        color: #e0e0e0 !important;
        font-style: italic;
    }

    /* Tables & dataframes */
    .stDataFrame, .stTable {
        background: rgba(0, 0, 0, 0.65) !important;
        color: white !important;
        border-radius: 8px;
    }

    /* Metrics */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #f9f9f9 !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
    }

    /* Titles */
    h1, h2, h3, h4, h5, h6 {
        color: #f9f9f9 !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
    }

    /* === ALERT / INFO / SUCCESS / WARNING / ERROR BOXES === */
    /* Info */
    .stAlert[data-testid="stInfo"] {
        background-color: rgba(0, 0, 0, 0.55) !important;
        color: white !important;
        border-radius: 10px;
    }
    .stAlert[data-testid="stInfo"] .stAlertContent svg {
        fill: white !important;
    }

    /* Success */
    .stAlert[data-testid="stSuccess"] {
        background-color: rgba(0, 128, 0, 0.55) !important;
        color: white !important;
        border-radius: 10px;
    }

    /* Warning */
    .stAlert[data-testid="stWarning"] {
        background-color: rgba(255, 165, 0, 0.55) !important;
        color: white !important;
        border-radius: 10px;
    }

    /* Error */
    .stAlert[data-testid="stError"] {
        background-color: rgba(255, 0, 0, 0.55) !important;
        color: white !important;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


import numpy as np
import pandas as pd
import joblib
import altair as alt
from sklearn.ensemble import GradientBoostingClassifier


# APP TITLE AND DESCRIPTION
col1, col2 = st.columns([1, 4])
with col1:
    st.image("background.png", width=80)
with col2:
    st.title("Bank Customer Churn Prediction App")
      # Horizontal separator line
st.markdown(
    "<hr style='height:3px;border:none;background-color:#333;' />",
    unsafe_allow_html=True)
    
    st.subheader("Single Prediction")


st.markdown(
    '<div style="background-color: rgba(0,0,0,0.55); color:white; padding:10px; border-radius:8px;">'
    'This app predicts whether a customer will churn using a trained Gradient Boosting model.'
    '</div>',
    unsafe_allow_html=True
)


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

    st.subheader("Customer Churn Patterns and Trends")

    st.write("""
    These charts help visualize how different customer characteristics relate to churn rate**.
    """)

    # Churn Rate by Age Group
    df["AgeGroup"] = pd.cut(df["Age"], bins=[18, 30, 40, 50, 60, 100],
                            labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    churn_by_age = df.groupby("AgeGroup")["Exited"].mean().reset_index()

    st.write("### 1. Churn Rate by Age Group")
    st.bar_chart(churn_by_age.set_index("AgeGroup"))
    st.caption("Older customers (especially 40+) tend to churn more often.")

    # Churn Rate by Account Balance Level
    df["BalanceGroup"] = pd.cut(df["Balance"], bins=[0, 50000, 100000, 150000, 200000, 250000],
                                labels=["0-50k", "50k-100k", "100k-150k", "150k-200k", "200k+"])
    churn_by_balance = df.groupby("BalanceGroup")["Exited"].mean().reset_index()

    st.write("### 2. Churn Rate by Account Balance")
    st.bar_chart(churn_by_balance.set_index("BalanceGroup"))
    st.caption("Customers with medium to high balances tend to leave more frequently.")

    # Churn Rate by Number of Products
    churn_by_products = df.groupby("NumOfProducts")["Exited"].mean().reset_index()

    st.write("### 3. Churn Rate by Number of Products")
    st.bar_chart(churn_by_products.set_index("NumOfProducts"))
    st.caption("Customers with only 1 product are more likely to churn, while those with 2 are more loyal.")

    # Churn Rate by Active Membership
    churn_by_active = df.groupby("IsActiveMember")["Exited"].mean().reset_index()
    churn_by_active["IsActiveMember"] = churn_by_active["IsActiveMember"].map({0: "Not Active", 1: "Active"})

    st.write("### 4. Churn Rate by Active Membership")
    st.bar_chart(churn_by_active.set_index("IsActiveMember"))
    st.caption("Inactive members have a much higher chance of leaving the bank.")

    # Churn Rate by Gender
    # Handle Gender column (if it's 0/1 numeric)
    if df["Gender"].dtype in [int, float]:
        df["Gender"] = df["Gender"].map({0: "Female", 1: "Male"})

    churn_by_gender = df.groupby("Gender")["Exited"].mean().reset_index()
    st.write("### 5. Churn Rate by Gender")
    st.bar_chart(churn_by_gender.set_index("Gender"))
    st.caption("There are slight differences in churn rate between male and female customers.")

    # Churn Rate by Country (handle one-hot encoded Geography)
    if "Geography" in df.columns:
        churn_by_geo = df.groupby("Geography")["Exited"].mean().reset_index()
    else:
        # Reconstruct pseudo Geography column from one-hot encoded columns
        df["Geography"] = "France"  # default baseline
        df.loc[df["Geography_Germany"] == 1, "Geography"] = "Germany"
        df.loc[df["Geography_Spain"] == 1, "Geography"] = "Spain"
        churn_by_geo = df.groupby("Geography")["Exited"].mean().reset_index()

    st.write("### 6. Churn Rate by Country")
    st.bar_chart(churn_by_geo.set_index("Geography"))
    st.caption("German customers show the highest churn rate, while French customers are the most stable.")


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
    st.markdown(
        f"""
        <div style="background-color: rgba(0,0,0,0.55); color:white; padding:10px; border-radius:8px;">
        Result = 1. Customer will churn.<br>Probability of churn: {prediction_proba:.2%}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <div style="background-color: rgba(0,0,0,0.55); color:white; padding:10px; border-radius:8px;">
        Result = 0. Customer will not churn.<br>Probability of churn: {prediction_proba:.2%}
        </div>
        """,
        unsafe_allow_html=True
    )


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


  # Horizontal separator line
st.markdown(
    "<hr style='height:3px;border:none;background-color:#333;' />",
    unsafe_allow_html=True
)
# ================================
# SECTION: DATASET PREDICTION
# ================================

st.header("Batch Prediction(DATASET)")

st.markdown(
    '<div style="background-color: rgba(0,0,0,0.55); color:white; padding:10px; border-radius:8px;">'
    'Upload a CSV file with customer data to predict churn for all customers using the trained model.'
    '</div>',
    unsafe_allow_html=True
)


# File uploader
st.markdown(
    """
    <div style="background-color: rgba(0,0,0,0.55); color:white; padding:15px; border-radius:8px;">
    <strong>Upload Customer Dataset (.csv)</strong>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded dataset
        user_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(user_df.head())

        # ======================
        # COLUMN ALIGNMENT
        # ======================
        feature_columns = df_train.drop("Exited", axis=1).columns

        # Reindex to ensure same columns as training data
        user_df = user_df.reindex(columns=feature_columns, fill_value=0)

        # Remove any unwanted or extra columns (e.g., from previous runs)
        user_df = user_df.loc[:, feature_columns]

        # ======================
        # PREDICTION SECTION
        # ======================
        predictions = model.predict(user_df)
        probabilities = model.predict_proba(user_df)[:, 1]

        # Add results back to the dataframe
        user_df["Churn_Prediction"] = predictions
        user_df["Churn_Probability"] = probabilities

        # Display predictions summary
        st.write("### Predictions Summary")
        st.dataframe(user_df.head())

        # Compute churn rate
        churn_rate = (user_df["Churn_Prediction"].mean()) * 100
        st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

        # ======================
        # VISUALIZATION SECTION
        # ======================
        st.subheader("Prediction Insights")

        # Churn vs Non-Churn Bar Chart
        churn_counts = user_df["Churn_Prediction"].value_counts().rename({0: "Not Churn", 1: "Churn"})
        st.bar_chart(churn_counts)

        # Age Distribution by Churn
        if "Age" in user_df.columns:
            st.write("**Age Distribution by Churn Status**")
            age_chart = alt.Chart(user_df).mark_boxplot().encode(
                x=alt.X("Churn_Prediction:N", title="Churn (0 = No, 1 = Yes)"),
                y=alt.Y("Age:Q", title="Customer Age"),
                color=alt.Color("Churn_Prediction:N", legend=None)
            ).properties(width=600)
            st.altair_chart(age_chart, use_container_width=True)

        # ======================
        # REPLACE THIS SECTION
        # ======================

        # Churn by Account Balance Group (instead of scatter plot)
        if "Balance" in user_df.columns:
            st.write("### Churn Rate by Account Balance Range")
            user_df["BalanceGroup"] = pd.cut(
                user_df["Balance"],
                bins=[0, 50000, 100000, 150000, 200000, 250000],
                labels=["0-50k", "50k-100k", "100k-150k", "150k-200k", "200k+"]
            )
            churn_by_balance_pred = (
                user_df.groupby("BalanceGroup")["Churn_Prediction"]
                .mean()
                .reset_index()
            )

            st.bar_chart(churn_by_balance_pred.set_index("BalanceGroup"))
            st.caption(
                "This shows the predicted churn rate across different account balance ranges. "
                "Higher churn among customers with very low or very high balances suggests "
                "these groups may need tailored retention strategies."
            )

        # Churn by Age Group
        if "Age" in user_df.columns:
            st.write("### Churn Rate by Age Group")
            user_df["AgeGroup"] = pd.cut(
                user_df["Age"],
                bins=[18, 30, 40, 50, 60, 100],
                labels=["18-30", "31-40", "41-50", "51-60", "60+"]
            )
            churn_by_age_pred = (
                user_df.groupby("AgeGroup")["Churn_Prediction"]
                .mean()
                .reset_index()
            )

            st.bar_chart(churn_by_age_pred.set_index("AgeGroup"))
            st.caption(
                "Older customers generally show a higher churn rate — "
                "highlighting the importance of loyalty programs for mature age segments."
            )

        # ======================
        # PATTERN IDENTIFICATION
        # ======================
        st.subheader("Pattern Insights")

        churned = user_df[user_df["Churn_Prediction"] == 1]
        non_churned = user_df[user_df["Churn_Prediction"] == 0]

        if not churned.empty and not non_churned.empty:
            avg_age_churn = churned["Age"].mean()
            avg_balance_churn = churned["Balance"].mean()
            avg_age_non = non_churned["Age"].mean()
            avg_balance_non = non_churned["Balance"].mean()

            pattern_text = f"""
            - Customers predicted to churn tend to have an **average age of {avg_age_churn:.1f}**, 
              compared to {avg_age_non:.1f} for those not likely to churn.  
            - Their **average balance** is around **${avg_balance_churn:,.0f}**, 
              compared to **${avg_balance_non:,.0f}** for non-churning customers.  
            - Overall churn rate in the uploaded dataset is **{churn_rate:.2f}%**.  
            """
            st.markdown(pattern_text)
        else:
            st.info("Not enough data variation to detect churn patterns.")

        # ======================
        # DOWNLOAD RESULTS
        # ======================
        st.subheader("Download Results")

        csv = user_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predicted Dataset as CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

        st.success("Dataset prediction and analysis complete!")

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

else:
    st.warning("Please upload a CSV file to generate churn predictions for your dataset.")
