import streamlit as st
import numpy as np
import pandas as pd
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



