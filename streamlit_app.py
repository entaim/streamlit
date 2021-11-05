import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
#from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  



digit_path = 'https://git.uwaterloo.ca/aaljmiai/ahdd1/-/raw/master/'
y_test = pd.read_csv(digit_path + 'csvTestLabel_10k_x_1.csv')
X_test = pd.read_csv(digit_path + 'csvTestImages_10k_x_784.csv')
#y_train = pd.read_csv(digit_path + 'csvTrainLabel_60k_x_1.csv')
#X_train = pd.read_csv(digit_path + 'csvTrainImages_60k_x_784.csv')

#dd = 
st.write(y_test.merge(X_test))

st.write("""
# Welcome to Booking  Prediction :star: 

""")
st.write('---')
st.write("""
# Dynamic Price Prediction 
* This app predicts the **Booking.com Prices**!

 

""")
st.write('---')


df4 = pd.read_csv('reg22.csv') 
x1= df4
x1['Log_price'] = np.log(x1['price'])

#st.write(x1)



X=  x1[['beds','number_of_ratings','rating']]
Y= x1['price']
#st.write(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=10)

#X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=10)

#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    
    #AGE = st.sidebar.slider('AGE',  float(X.AGE.min()),  float(X.AGE.max()),  float(X.AGE.mean()))
    beds = st.sidebar.slider('Number of beds',  int(X.beds.min()),  int(X.beds.max()),  int(X.beds.mean()))
    review = st.sidebar.slider('Number of reviews',  int(X.number_of_ratings.min()),  int(X.number_of_ratings.max()),  int(X.number_of_ratings.mean()))
    rating = st.sidebar.slider('Ratings',  float(X.rating.min()),  float(X.rating.max()),  float(X.rating.mean()))
   # Size = st.sidebar.slider('Room Size(m2)',  float(X.Size.min()),  float(X.Size.max()),  float(X.Size.mean()))
   
   
    data = {'Number of beds': beds,
            'Number of reviews': review,
            'Ratings': rating,
            }

            
    features = pd.DataFrame(data, index=[0])
    return features
  
df = user_input_features()

# Main Panel

# Print specified input parameters
#st.header('Specified Input parameters')
#st.write()
#st.write('---')

st.set_option('deprecation.showPyplotGlobalUse', False)
# Build Regression Model
model = RandomForestRegressor(n_estimators=10, random_state=0)
model.fit(X_train, y_train)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Predicted Price (Saudi Riyal) :red_circle:')
st.write(round(prediction[0], 2),"SR") 
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
st.write('* SHAPE values show how much a given feature changed our prediction')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')


plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')

st.write('---')
st.write('## Our Client :dizzy:')
#st.write(x1)
#st.write()
st.write("""



![](https://user-images.githubusercontent.com/20365333/138615337-bfbdfdb2-494c-4265-8ff0-467b158f95d3.png)


""")





