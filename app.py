import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import *


st.title(" Water Quality **Estimation**")


  <img width="523" src="https://user-images.githubusercontent.com/20365333/140042600-a602ed75-6571-4f7b-adee-0d33d51f9cf0.jpg">


st.write("""



![](https://user-images.githubusercontent.com/20365333/140042600-a602ed75-6571-4f7b-adee-0d33d51f9cf0.jpg)



""")

st.markdown("""
	Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection.
	This is important as a health and development issue at a national, regional and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit,
	since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.
	""")

st.subheader("sample values for the input")
df=pd.read_csv("example.csv")
#df.drop('Unnamed: 0', axis=1, inplace=True)
df

#df_example=df.iloc[df['ph']==6.007427,['Organic_carbon','Conductivity','Hardness']]
#df_example=df.iloc[0,1:]
#df_example

if st.checkbox("Show orignal dataframe"):
	dataframe=pd.read_csv("water1.csv")
	dataframe.drop('Unnamed: 0', axis=1, inplace=True)
	dataframe

##Sidebar

st.sidebar.title("Select your Input  Values")

uploaded_file=st.sidebar.file_uploader("Upload your csv file in the same input as the example csv file",type=["csv"])

if uploaded_file is not None:
	input_params=pd.read_csv(uploaded_file)

else:
	ph=st.sidebar.slider("ph value",2.1,28.3,12.5)
	Hardness=st.sidebar.slider("Hardness value",47.432,323.3,158.2)
	Solids=st.sidebar.slider("Solids value",181.4,753.2,442.85)
	Chloramines=st.sidebar.slider("Chloramines value",2.1,28.3,12.5)
	Sulfate=st.sidebar.slider("Sulfate value",47.432,323.3,158.2)
	Conductivity=st.sidebar.slider("Conductivity value",181.4,753.2,442.85)
	Organic_carbon=st.sidebar.slider("Organic Carbon value",2.1,28.3,12.5)
	Trihalomethanes=st.sidebar.slider("Trihalomethanesvalue",47.432,323.3,158.2)
	Turbidity=st.sidebar.slider("Turbidity value",2.1,28.3,12.5)

	dict_values={"ph":ph, "Hardness":Hardness, "Solids":Solids,"Chloramines":Chloramines,"Sulfate":Sulfate,
		     "Conductivity":Conductivity,"Organic_carbon":Organic_carbon,
		     "Trihalomethanes":Trihalomethanes,"Turbidity":Turbidity}
	features=pd.DataFrame(dict_values,index=[0])
	input_params=features



#ph Solids Chloramines Sulfate Trihalomethanes Turbidity

st.subheader("user input fields")

if uploaded_file is not None:
	st.write(input_params)
else:
	st.write(input_params)

#load_clf=pickle.load(open('dt_saved_07032020.pkl','rb'))
#load_clf= pd.read_csv("water1.csv")
load_clf= load_model('dt_saved_07032020')
prediction=load_clf.predict(input_params)
st.subheader("The Prediction is")
st.write(prediction[0])

if(prediction[0]==1):
	st.subheader("The water is safe to drink")
else:
	st.subheader("The water is not safe to drink")
