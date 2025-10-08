import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt
from sklearn.ensemble import GradientBoostingClassifier


# APP TITLE AND DESCRIPTION

st.title('Page TWO')
st.info('This page predicts whether a group of customer will churn using a trained Gradient Boosting model.')
