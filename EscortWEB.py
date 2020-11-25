#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor 



@st.cache
def load_data():
    df = pd.read_csv('out.csv')
    return df

df = load_data()
data = df[['Age', 'Boobs', 'Height', 'Size', 'Weight']]
target = pd.DataFrame(df['Price_USD'])

hgbr = HistGradientBoostingRegressor()
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=42, test_size=0.33)
hgbr.fit(x_train, y_train)

st.header('Сколько ты стоишь?')
st.subheader('Ты можешь оценить свою **часовую оплату** на основе своих параметров. ')

st.sidebar.header('Введи свои параметры')
age = st.sidebar.slider('Возвраст', min_value=18, max_value=50, value=18, step=1, key='age')
boobs = st.sidebar.slider('Размер груди', min_value=1, max_value=7, value=1, step=1, key='boobs')
height = st.sidebar.slider('Рост, см', min_value=120, max_value=200, value=165, key='height')
size = st.sidebar.slider('Размер одежды',  min_value=38, max_value=56, value=42, step=2, key='size')
weight = st.sidebar.slider('Вес, кг', min_value=40, max_value=150, value=45, key='weight')

slut = np.array([age, boobs, height, size, weight]).reshape((1,-1))
cost_slut = hgbr.predict(slut)

st.subheader('Ваша стоимость в долларах')
st.write(cost_slut)

if cost_slut < 50:
    st.subheader('К сожалению, вы относитесь к категории "Дешевые шлюхи"')
    
if cost_slut < 100 and cost_slut >= 50:
    st.subheader('Не так уж и плохо, вы относитесь к категории "Средние шлюшки"')    

if cost_slut >= 100:
    st.subheader('Позравляем, вы относитесь к категории "Элитные эскортницы"')

