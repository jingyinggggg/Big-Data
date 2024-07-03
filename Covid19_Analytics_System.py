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

# 2 ------------------------- Weekly Confirmed Chart ---------------------------------------------

def weekly_confirmed_chart():
    weekly_confirmed_data = date_filter.set_index('Date').resample('W').sum().reset_index()
    fig = px.line(weekly_confirmed_data, x='Date', y='Confirmed', title='Weekly Confirmed Cases')
    st.plotly_chart(fig)

def actual_weekly_confirmed_chart_2020():
    weekly_confirmed_data_2020 = cases_data_2020.set_index('Date').resample('W').sum().reset_index()
    fig = px.line(weekly_confirmed_data_2020, x='Date', y='Confirmed', title='Actual Weekly Confirmed Cases for 2020')
    st.plotly_chart(fig)

def predict_weekly_confirmed():
    weekly_confirmed_data = date_filter.set_index('Date').resample('W').sum().reset_index()
    weekly_confirmed_data['WeekOfYear'] = weekly_confirmed_data['Date'].dt.isocalendar().week
    X = weekly_confirmed_data[['WeekOfYear']]
    y = weekly_confirmed_data['Confirmed']

    model = LinearRegression()
    model.fit(X, y)

    # Create a DataFrame for 2021 prediction
    weeks_in_2021 = np.arange(1, 53).reshape(-1, 1)
    predicted_confirmed_2021 = model.predict(weeks_in_2021)
    predicted_data_2021 = pd.DataFrame({
        'Week': weeks_in_2021.flatten(),
        'Predicted Confirmed': predicted_confirmed_2021
    })

    fig = px.bar(predicted_data_2021, x='Week', y='Predicted Confirmed', title='Predicted Weekly Confirmed Cases for 2021')
    st.plotly_chart(fig)

def descriptive_analysis_confirmed():
    st.subheader("Descriptive Analysis of Weekly Confirmed Cases:")
    st.write(date_filter['Confirmed'].describe())
    
    st.subheader("Box Plot of Weekly Confirmed Cases")
    fig = px.box(date_filter, y='Confirmed', title='Box Plot of Weekly Confirmed Cases')
    st.plotly_chart(fig)

# 3 ------------------------- Weekly Recovered Chart ---------------------------------------------

def weekly_recovered_chart():
    weekly_recovered_data = date_filter.set_index('Date').resample('W').sum().reset_index()
    fig = px.line(weekly_recovered_data, x='Date', y='Recovered', title='Weekly Recovered Cases')
    st.plotly_chart(fig)
    
def actual_weekly_recovered_chart_2020():
    weekly_recovered_data_2020 = cases_data_2020.set_index('Date').resample('W').sum().reset_index()
    fig = px.line(weekly_recovered_data_2020, x='Date', y='Recovered', title='Actual Weekly Recovered Cases for 2020')
    st.plotly_chart(fig)

def predict_weekly_recovered():
    weekly_recovered_data = date_filter.set_index('Date').resample('W').sum().reset_index()
    weekly_recovered_data['WeekOfYear'] = weekly_recovered_data['Date'].dt.isocalendar().week
    X = weekly_recovered_data[['WeekOfYear']]
    y = weekly_recovered_data['Recovered']

    model = LinearRegression()
    model.fit(X, y)

    # Create a DataFrame for 2021 prediction
    weeks_in_2021 = np.arange(1, 53).reshape(-1, 1)
    predicted_recovered_2021 = model.predict(weeks_in_2021)
    predicted_data_2021 = pd.DataFrame({
        'Week': weeks_in_2021.flatten(),
        'Predicted Recovered': predicted_recovered_2021
    })

    fig = px.bar(predicted_data_2021, x='Week', y='Predicted Recovered', title='Predicted Weekly Recovered Cases for 2021')
    st.plotly_chart(fig)

def descriptive_analysis_recovered():
    descriptive_stats = date_filter['Recovered'].describe()
    st.write("Descriptive Analysis of Weekly Recovered Cases:")
    st.write(descriptive_stats)

# 4 ------------------------- Total death vs cases chart ---------------------------------------------

def total_deaths_vs_cases_chart():
    total_data = cases_data.groupby('Country').agg({'Deaths': 'sum', 'Confirmed': 'sum'}).reset_index()
    fig = px.scatter(total_data, x='Confirmed', y='Deaths', color='Country', 
                     size='Deaths', hover_name='Country', log_x=True, log_y=True,
                     title='Total Deaths vs Cases by Country')
    st.plotly_chart(fig)

# # 5 ------------------------- Case by Country ---------------------------------------------

def cases_by_country_chart():
    cases_by_country_data = cases_data.groupby('Country')['Confirmed'].sum().reset_index()
    fig = px.choropleth(cases_by_country_data, locations='Country', locationmode='country names', 
                        color='Confirmed', hover_name='Country', 
                        title='Number of Cases by Country',
                        color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig)
