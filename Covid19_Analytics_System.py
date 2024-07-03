import streamlit as st  # For the dashboard
import pandas as pd  # To read data
import plotly.express as px  # To create charts
from streamlit_option_menu import option_menu  # To handle menu option
from sklearn.linear_model import LinearRegression  # For linear regression
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # For ARIMA model

# Define chart functions
# 1 ------------------------- Death Rate Chart ---------------------------------------------

# Predicting 2021 Death Rate
# Assuming daily data and predicting using Linear Regression
cases_data_2020['DayOfYear'] = cases_data_2020['Date'].dt.dayofyear
X = cases_data_2020[['DayOfYear']]
y = cases_data_2020['Deaths']

model = LinearRegression()
model.fit(X, y)

# Create a DataFrame for 2021 prediction
days_in_2021 = np.arange(1, 366).reshape(-1, 1)
predicted_deaths_2021 = model.predict(days_in_2021)
predicted_data_2021 = pd.DataFrame({
    'Date': pd.date_range(start='2021-01-01', end='2021-12-31'),
    'Predicted Deaths': predicted_deaths_2021
})

def death_rate_chart():
    filtered_data = date_filter.groupby('Date').sum().reset_index()
    fig = px.area(filtered_data, x='Date', y='Deaths', title='Cumulative Death Rates Over Time', 
                  labels={'Deaths': 'Cumulative Deaths'})
    st.plotly_chart(fig)
    
def actual_death_rate_chart():
    filtered_data = cases_data_2020.groupby('Date').sum().reset_index()
    filtered_data['Death Rate'] = filtered_data['Deaths']
    fig = px.bar(filtered_data, x='Date', y='Death Rate', title='Actual Death Rate Over Time (2020)')
    st.plotly_chart(fig)

def predicted_death_rate_chart():
    # Aggregate daily death counts to ensure unique dates
    daily_deaths = cases_data_2020.groupby('Date')['Deaths'].sum()
    
    # Fit ARIMA model
    daily_deaths = daily_deaths.asfreq('D').fillna(0)  # Ensure the series has a daily frequency
    model = ARIMA(daily_deaths, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast for 2021
    forecast = model_fit.forecast(steps=365)
    forecast_dates = pd.date_range(start='2021-01-01', end='2021-12-31')
    forecast_data = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted Deaths': forecast.values
    })

    fig = px.line(forecast_data, x='Date', y='Predicted Deaths', title='Predicted Death Rate for 2021')
    st.plotly_chart(fig)

def descriptive_analysis_chart():
    st.subheader("Descriptive Analysis of Death Rate (2020):")
    st.write(cases_data_2020['Deaths'].describe())
    
    st.subheader("Box Plot of Death Rates")
    fig = px.box(cases_data_2020, y='Deaths', title='Box Plot of Death Rates in 2020')
    st.plotly_chart(fig)
