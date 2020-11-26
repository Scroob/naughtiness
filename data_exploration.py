#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler

def load_page(df):
    prepare_layout()
    some_plot(df)
#     plot_play_count_graph(df)
#     longest_break_between_games(df)
#     most_subsequent_days_played(df)
#     most_games_on_one_day(df)

def prepare_layout():
    '''
    ### Проверка МаркДауна
    ## Еще *одна* **проверка**
    '''
    st.markdown("There are several things you see on this page:".format(SPACES))
    st.markdown("{}🔹 On the **left** you can see how often games were played "
                "in the last year of matches. ".format(SPACES))
    st.markdown("{}🔹 You can see the **total amount** certain board games have been played. ".format(SPACES))
    st.markdown("{}🔹 The longest **break** between board games. ".format(SPACES))
    st.markdown("{}🔹 The **longest chain** of games played in days. ".format(SPACES))
    st.markdown("{}🔹 The **day** most games have been played. ".format(SPACES))
    st.write(" ")

def some_plot(df):
    
    ages = sorted(df['Age'].unique())
    boobs_avg = np.array([np.mean(df[df['Age']== x]['Boobs']) for x in ages])
    weight_avg = [np.mean(df[df['Age']== x]['Weight']) for x in ages]

    scaler = MinMaxScaler()
    boobs_avg_scaled = scaler.fit_transform((boobs_avg).reshape((-1,1)))
    
    fig = px.scatter(x=ages, y=weight_avg, size=boobs_avg_scaled, size_max=60)
    st.write(fig)
