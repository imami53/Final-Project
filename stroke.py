import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
#import joblib
import warnings
warnings.filterwarnings('ignore')
#import plotly.express as px

data = pd.read_csv('healthcare-dataset-stroke-data (1).csv')

st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Bazooka'>STROKE PREDICTION</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>BUILT BY OGECHI EKENE</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.image('pngwing.com (4).png')
st.divider()

st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown("Stroke is a leading cause of disability and death worldwide, resulting from the interruption of blood flow to the brain due to a clot (ischemic stroke) or bleeding (hemorrhagic stroke). Early prediction of stroke risk is crucial for timely intervention and prevention.Recent advancements in machine learning and artificial intelligence have enabled the development of predictive models using healthcare data. These models leverage clinical parameters such as age, hypertension, heart disease, smoking status, and lifestyle factors to assess an individual's stroke risk. Datasets, such as those available on Kaggle, provide valuable structured data for training and evaluating predictive models.")


st.divider()

st.dataframe(data, use_container_width = True)

st.sidebar.image('pngwing.com (5).png', caption = 'Welcome User')

#['avg_glucose_level','bmi','age','smoking_status','heart_disease','gender','work_type','hypertension', 'stroke']

glucose = st.sidebar.number_input('Avg_glucose', min_value=0.0, max_value=10000.0, value=data.avg_glucose_level.median())
bmi = st.sidebar.number_input('Body mass index', min_value=0.0, max_value=10000.0, value=data.bmi.median())
age = st.sidebar.number_input('Age', min_value= 0.0, max_value = 10000.0, value=data.age.median())
smoking = st.sidebar.selectbox('Smoking_status', data.smoking_status.unique(),index=1)
heart = st.sidebar.selectbox('Heart_disease_status', data.heart_disease.unique(),index=1)
gender = st.sidebar.selectbox('Gender', data.gender.unique(),index=1)
work = st.sidebar.selectbox('Work_Type', data.work_type.unique(),index=1)
hypertension = st.sidebar.selectbox('Hypertension_status', data.hypertension.unique(),index=1)

inputs = {
    'avg_glucose_level' : [glucose],
    'bmi' : [bmi],
    'age' : [age],
    'smoking_status' : [smoking],
    'heart_disease' : [heart],
    'gender' : [gender],
    'work_type' : [work],
    'hypertension' : [hypertension]
     }

inputVar = pd.DataFrame(inputs)
st.divider()
st.header('User Input')
st.dataframe(inputVar)

# transforming the user input, input the transformers
smoking_encoder = joblib.load('smoking_status_encoder.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
work_encoder = joblib.load('work_type_encoder.pkl')


# use the imported transformer to transform user input
inputVar['smoking_status'] = smoking_encoder.transform(inputVar[['smoking_status']])
inputVar['gender'] = gender_encoder.transform(inputVar[['gender']])
inputVar['work_type'] = work_encoder.transform(inputVar[['work_type']])

model = joblib.load('stroke_prediction_model.pkl')
predict = model.predict(inputVar)

if  st.button('Check Your Stroke Risk'):
    if predict == 1:
        st.error('Unfortunately, You have a high risk of having a stroke')
    else:
        st.success('Congratulations, You have a low risk of having a stroke')



