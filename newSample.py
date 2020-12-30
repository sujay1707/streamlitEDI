import streamlit as st 
import streamlit.components.v1 as stc
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import altair as alt
import numpy as np 

df=pd.read_csv(r'C:\Users\sujay\OneDrive\Documents\PythonPrj\c21.csv')

nav=st.sidebar.radio('NAVIGATION',['Home','Inventory'])

if nav=='Home':
    HtmlFile = open("design.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    stc.html(source_code)
    st.image("pharmacy.jpg",width=700)
    if st.checkbox("Show Table"):
        st.table(df)
    
    
if nav=='Inventory':
    HtmlFile = open("design.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    stc.html(source_code)
    df=df.dropna()
    train=df.iloc[:-30]
    test=df.iloc[-30:]
    model=sm.tsa.statespace.SARIMAX(train['AvgTemp'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
    model=model.fit()
    model.summary()
    start=len(train)
    end=len(train)+len(test)-1
    #if the predicted values dont have date values as index, you will have to uncomment the following two commented lines to plot a graph
    #index_future_dates=pd.date_range(start='2018-12-01',end='2018-12-30')
    pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA predictions')
    #pred.index=index_future_dates
    

    
    pred.plot(legend='ARIMA Predictions')
    test['AvgTemp'].plot(legend=True)
    test['AvgTemp'].mean()
    rmse=sqrt(mean_squared_error(pred,test['AvgTemp']))
    print(rmse)
    model2=sm.tsa.statespace.SARIMAX(df['AvgTemp'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
    model2=model2.fit()
    index_future_dates=pd.date_range(start='1972-10-01',end='1975-04-01',freq='MS')
    #print(index_future_dates)
    pred=model2.predict(start=len(df),end=len(df)+30,typ='levels').rename('ARIMA Predictions')
    #print(comp_pred)
    pred.index=index_future_dates
    print(pred)
    st.line_chart(pred)