import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import load_boston


st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')

#boston = load_boston()
#from sklearn.datasets import load_boston
boston = load_boston()
#X, Y = boston.data, boston.target
#print(boston)
#boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
#boston['MEDV'] = boston_dataset.target
# Loads the Boston House Price Dataset
#boston = "http://lib.stat.cmu.edu/datasets/boston"
#X = pd.DataFrame(raw_df.data, columns=raw_df.feature_names)
#Y = pd.DataFrame(raw_df.target, columns=["MEDV"])
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

Y.head()
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')
