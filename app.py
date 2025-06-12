import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load the model , pickle files

model=load_model('model.h5')

with open ('gender_value.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('geo_value.pkl','rb') as file:
    one_hot_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


# initiate Streamlit app

st.title('Bank Customer Churn predictor')


# user input

CreditScore = st.number_input('Credit Score')
Gender	= st.selectbox('Gender',label_encoder_gender.classes_)
Age	= st.slider('Age',18,92)
Tenure	= st.slider('Tenure',0,10)
Balance	= st.number_input('Balance')
NumOfProducts = st.slider('Number of products eg., credit card, car loan', 0,4)
HasCrCard= st.selectbox('Has Credit card',[0,1])
IsActiveMember	= st.selectbox('Active member',[0,1])
EstimatedSalary	= st.number_input ('Estimated salary')
Geography = st.selectbox('Geography', one_hot_geo.categories_[0])



# prepare the input data

input_data= pd.DataFrame({    
    'CreditScore' : [CreditScore],
    'Gender': label_encoder_gender.transform([Gender])[0],
    'Age':Age,
    'Tenure':Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': IsActiveMember,
    'EstimatedSalary': EstimatedSalary,
    })


# treating geography data

geo_encoded=one_hot_geo.transform([[Geography]]).toarray()

geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_geo.get_feature_names_out())

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# scaling data

input_data_scaled=scaler.transform(input_data)

# feed into model to predict

prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

st.write(f'Churn probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write("Yes customer gonna churn")
else:
    st.write("Customer is NOT likely to churn")