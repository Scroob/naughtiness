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
    st.markdown("На этой странице мы приводим предварительный анализ данных)
    st.write(" ")

def some_plot(df):
    
    ages = sorted(df['Age'].unique())
    boobs_avg = np.array([np.mean(df[df['Age']== x]['Boobs']) for x in ages])
    weight_avg = [np.mean(df[df['Age']== x]['Weight']) for x in ages]

    scaler = MinMaxScaler()
    boobs_avg_scaled = scaler.fit_transform((boobs_avg).reshape((-1,1)))
    
    fig = px.scatter(x=ages, y=weight_avg, size=boobs_avg_scaled, size_max=60)
    st.write(fig)
