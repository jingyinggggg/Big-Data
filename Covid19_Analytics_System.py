import streamlit as st  # For the dashboard
import pandas as pd  # To read data
import plotly.express as px  # To create charts
from streamlit_option_menu import option_menu  # To handle menu option
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # For ARIMA model
from pymongo import MongoClient # To connect to MongoDB
from sklearn.ensemble import RandomForestRegressor
import time

# Page configuration
st.set_page_config(page_title="Covid-19 Analytics", page_icon=":syringe:", layout="wide")

# Load CSS Style
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Load datasets
    start_time = time.time()
    client = MongoClient("mongodb://localhost:27017")
    db = client["vaccine_analysis"]
    
    cases_data = pd.DataFrame(list(db["aggregated_data_cleaned"].find()))
    vaccine_data = pd.DataFrame(list(db["vaccine_data_cleaned"].find()))
    end_time = time.time()
    print(f"Data loading time: {end_time - start_time} seconds")

    # Process Data for Analysis
    start_time = time.time()
    cases_data['Date'] = pd.to_datetime(cases_data['Date'], format='%Y-%m-%d', errors='coerce')
    vaccine_data['date'] = pd.to_datetime(vaccine_data['date'], format='%Y-%m-%d', errors='coerce')
    end_time = time.time()
    print(f"Data processing time: {end_time - start_time} seconds")
    
    return cases_data, vaccine_data 

@st.cache_data
def preprocess_data(cases_data):
    cases_data['Date'] = pd.to_datetime(cases_data['Date'], format='%Y-%m-%d', errors='coerce')
    cases_data['Year'] = cases_data['Date'].dt.year
    return cases_data[cases_data['Year'] == 2020].copy()

# Load and process data
cases_data, vaccine_data = load_data()
cases_data_2020 = preprocess_data(cases_data)

# Extract year from date
cases_data['Year'] = cases_data['Date'].dt.year
vaccine_data['Year'] = vaccine_data['date'].dt.year

##########################################

# Sidebar Filters
st.sidebar.header("Please Filter Here:")

# Year filter
available_years = cases_data["Year"].dropna().unique()
selected_years = st.sidebar.multiselect("Select the Year:", options=available_years)

# Country filter
available_countries = cases_data["Country"].unique()
selected_countries = st.sidebar.multiselect("Select the Country:", options=available_countries)

# Filter data based on sidebar input
start_time = time.time()
year_filter = cases_data[cases_data["Year"].isin(selected_years)] if selected_years else cases_data
country_filter = year_filter[year_filter["Country"].isin(selected_countries)] if selected_countries else year_filter
country_filter2020 = cases_data_2020[cases_data_2020["Country"].isin(selected_countries)] if selected_countries else cases_data_2020
end_time = time.time()
print(f"Sidebar filtering time: {end_time - start_time} seconds")

# Date range filter
min_date = cases_data["Date"].min().to_pydatetime()
max_date = cases_data["Date"].max().to_pydatetime()
selected_date_range = st.sidebar.slider("Select the Date Range:", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD")

date_filter = country_filter[(country_filter["Date"] >= selected_date_range[0]) & (country_filter["Date"] <= selected_date_range[1])].copy()

# Define sidebar
with st.sidebar:
    menu_choice = option_menu("Main Menu", ["Home", 'Death Rate Charts', "Weekly Confirmed Case", 'Weekly Recovered Case', 'Cases by Country',
                                            'Total Death VS Cases', 'Vaccine by Country', 'Effectiveness of Vaccines'], 
        icons=['house', 'graph-down', 'virus', 'bandaid', 'globe-americas','heart-pulse', 'journals', 'clipboard-heart'], menu_icon="cast", default_index=1)

# Define chart functions
# 1 ------------------------- Death Rate Chart ---------------------------------------------

def profile(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {execution_time:.4f} seconds")
        return result
    return wrapper

@profile
def death_rate_chart():
    filtered_data = date_filter.groupby('Date')[date_filter.select_dtypes(include='number').columns].sum().reset_index()

    fig = px.area(filtered_data, x='Date', y='Deaths', title='Cumulative Death Rates Over Time', 
                  labels={'Deaths': 'Cumulative Deaths'})
    st.plotly_chart(fig)

@profile    
def actual_death_rate_chart():
    filtered_data = country_filter2020.groupby('Date')[country_filter2020.select_dtypes(include='number').columns].sum().reset_index()
    filtered_data['Death Rate'] = filtered_data['Deaths']
    fig = px.bar(filtered_data, x='Date', y='Death Rate', title='Actual Death Rate Over Time (2020)')
    st.plotly_chart(fig)

@profile
def predicted_death_rate_chart():
    # Aggregate daily death counts to ensure unique dates
    daily_deaths = country_filter2020.groupby('Date')['Deaths'].sum()

    # Ensure the series has a daily frequency, fill missing dates with NaN first
    daily_deaths = daily_deaths.asfreq('D').fillna(0)

    # Check if the data is stationary
    stationary = is_stationary(daily_deaths)
    if not stationary:
        # Difference the data to make it stationary
        daily_deaths_diff = daily_deaths.diff().dropna()
    else:
        daily_deaths_diff = daily_deaths

    # Fit ARIMA model
    model = ARIMA(daily_deaths_diff, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast for 2021
    forecast_steps = 365  # Number of days to forecast
    forecast_diff = model_fit.forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(start='2021-01-01', periods=forecast_steps, freq='D')

    # Convert forecast from differenced to original scale
    if not stationary:
        # Inverse differencing to get the forecast back to original scale
        forecast = pd.Series(forecast_diff).cumsum() + daily_deaths.iloc[-1]
    else:
        forecast = forecast_diff

    # Create DataFrame for forecast results
    forecast_data = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted Deaths': forecast
    })

    # Plot forecast data
    fig = px.line(forecast_data, x='Date', y='Predicted Deaths', title='Predicted Death Rate for 2021')
    st.plotly_chart(fig)

def is_stationary(data):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(data)
    return result[1] <= 0.05  # p-value less than or equal to 0.05 indicates stationarity

@profile
def descriptive_analysis_chart():
    st.subheader("Descriptive Analysis of Death Rate (2020):")
    st.write(country_filter2020['Deaths'].describe())

# 2 ------------------------- Weekly Confirmed Chart ---------------------------------------------
@profile
def weekly_confirmed_chart():
    weekly_confirmed_data = date_filter.set_index('Date')['Confirmed'].resample('W').sum().reset_index()
    fig = px.line(weekly_confirmed_data, x='Date', y='Confirmed', title='Weekly New Cases', labels={'Confirmed': 'Cumulative Weekly Cases'})
    st.plotly_chart(fig)

@profile
def actual_weekly_confirmed_chart_2020():
    weekly_confirmed_data_2020 = cases_data_2020.set_index('Date')['Confirmed'].resample('W').sum().reset_index()
    fig = px.line(weekly_confirmed_data_2020, x='Date', y='Confirmed', title='Actual Weekly New Cases for 2020', labels={'Confirmed': 'Cumulative Weekly Cases'})
    st.plotly_chart(fig)

@profile
def predict_weekly_confirmed():
    # Resample daily data to weekly and sum
    weekly_confirmed_data = date_filter.set_index('Date')['Confirmed'].resample('W').sum().reset_index()
    weekly_confirmed_data['WeekOfYear'] = weekly_confirmed_data['Date'].dt.isocalendar().week
    X = weekly_confirmed_data[['WeekOfYear']]
    y = weekly_confirmed_data['Confirmed']

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create a DataFrame for 2021 prediction
    weeks_in_2021 = np.arange(1, 53).reshape(-1, 1)
    X_predict = pd.DataFrame(weeks_in_2021, columns=['WeekOfYear'])
    predicted_confirmed_2021 = model.predict(X_predict)
    # predicted_confirmed_2021 = model.predict(weeks_in_2021)
    predicted_data_2021 = pd.DataFrame({
        'Week': weeks_in_2021.flatten(),
        'Predicted New Cases': predicted_confirmed_2021
    })

    # Plot predicted weekly new cases for 2021 using Plotly Express
    fig = px.line(predicted_data_2021, x='Week', y='Predicted New Cases', title='Predicted Weekly New Cases for 2021')
    st.plotly_chart(fig)

@profile
def descriptive_analysis_confirmed():
    weekly_confirmed_data = date_filter.set_index('Date')['Confirmed'].resample('W').sum().reset_index()
    
    st.subheader("Descriptive Analysis of Weekly New Cases:")
    st.write(weekly_confirmed_data['Confirmed'].describe())
    
    st.subheader("Box Plot of Weekly New Cases")
    fig = px.box(weekly_confirmed_data, y='Confirmed', title='Box Plot of Weekly New Cases', labels={'Confirmed': 'Cumulative Weekly Cases'})
    st.plotly_chart(fig)

# 3 ------------------------- Weekly Recovered Chart ---------------------------------------------

# Function to convert dates to the end of the week
def get_week_end(date):
    return date + pd.DateOffset(days=(6 - date.weekday()))

# Add a new column to hold the week end date
date_filter['Week_End'] = date_filter['Date'].apply(get_week_end)

@profile
def weekly_recovered_chart():
    # Group by the week end date and sum the recovered cases
    weekly_recovered_data = date_filter.groupby('Week_End')['Recovered'].sum().reset_index()
    
    fig = px.line(weekly_recovered_data, x='Week_End', y='Recovered', title='Weekly Recovered Cases', labels={'Recovered': 'Cumulative Weekly Recovered Cases'})
    st.plotly_chart(fig)

@profile
def actual_weekly_recovered_chart_2020():
    # Filter the data for the year 2020
    data_2020 = date_filter[date_filter['Date'].dt.year == 2020]
    
    # Group by the week end date and sum the recovered cases
    weekly_recovered_data_2020 = data_2020.groupby('Week_End')['Recovered'].sum().reset_index()
    
    fig = px.line(weekly_recovered_data_2020, x='Week_End', y='Recovered', title='Actual Weekly Recovered Cases for 2020', labels={'Recovered': 'Cumulative Weekly Recovered Cases'})
    st.plotly_chart(fig)

@profile
def predict_weekly_recovered():
    # Group by the week end date and sum the recovered cases
    weekly_recovered_data = date_filter.groupby('Week_End')['Recovered'].sum().reset_index()
    
    weekly_recovered_data['WeekOfYear'] = weekly_recovered_data['Week_End'].dt.isocalendar().week
    X = weekly_recovered_data[['WeekOfYear']]
    y = weekly_recovered_data['Recovered']

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create a DataFrame for 2021 prediction
    weeks_in_2021 = np.arange(1, 53).reshape(-1, 1)
    X_predict = pd.DataFrame(weeks_in_2021, columns=['WeekOfYear'])
    predicted_recovered_2021 = model.predict(X_predict)
    # predicted_recovered_2021 = model.predict(weeks_in_2021)
    predicted_data_2021 = pd.DataFrame({
        'Week': weeks_in_2021.flatten(),
        'Predicted Recovered': predicted_recovered_2021
    })

    # Plot predicted weekly recovered cases for 2021 using Plotly Express
    fig = px.line(predicted_data_2021, x='Week', y='Predicted Recovered', title='Predicted Weekly Recovered Cases for 2021')
    st.plotly_chart(fig)

@profile
def descriptive_analysis_recovered():
    # Group by the week end date and sum the recovered cases
    weekly_recovered_data = date_filter.groupby('Week_End')['Recovered'].sum().reset_index()
    
    descriptive_stats = weekly_recovered_data['Recovered'].describe()
    st.write("Descriptive Analysis of Weekly Recovered Cases:")
    st.write(descriptive_stats)

        
# 4 ------------------------- Total death vs cases chart ---------------------------------------------
@profile
def total_deaths_vs_cases_chart(cases_data):
    # Group data
    total_data = cases_data.groupby('Country').agg({'Daily_Deaths': 'sum', 'Daily_New_Cases': 'sum'}).reset_index()
    
    # Convert to numeric, handle errors
    total_data['Daily_Deaths'] = pd.to_numeric(total_data['Daily_Deaths'], errors='coerce')
    total_data['Daily_New_Cases'] = pd.to_numeric(total_data['Daily_New_Cases'], errors='coerce')
    
    # Drop NaN values
    total_data = total_data.dropna(subset=['Daily_Deaths', 'Daily_New_Cases'])
    
    # Filter out zero or negative values
    total_data = total_data[(total_data['Daily_New_Cases'] > 0) & (total_data['Daily_Deaths'] > 0)]
    
    # Get the minimum and maximum values for the axes
    min_confirmed = max(total_data['Daily_New_Cases'].min(), 1)
    max_confirmed = total_data['Daily_New_Cases'].max()
    min_deaths = max(total_data['Daily_Deaths'].min(), 1)
    max_deaths = total_data['Daily_Deaths'].max()
    
    # Create a scatter plot
    fig = px.scatter(total_data, x='Daily_New_Cases', y='Daily_Deaths', color='Country', labels={'Daily_New_Cases': 'Cases', 'Daily_Deaths': 'Total deaths'},
                     size='Daily_Deaths', hover_name='Country', log_x=True, log_y=True,
                     title='Total Deaths vs Cases by Country',
                     range_x=[min_confirmed, max_confirmed],
                     range_y=[min_deaths, max_deaths])
    
    # Display the chart
    st.plotly_chart(fig)

@profile
def descriptive_analysis_deaths():
    total_data = cases_data.groupby('Country').agg({'Daily_Deaths': 'sum', 'Daily_New_Cases': 'sum'}).reset_index()
    descriptive_stats_deaths = total_data['Daily_Deaths'].describe()
    st.write("Descriptive Analysis of Total Deaths:")
    st.write(descriptive_stats_deaths)

@profile
def descriptive_analysis_cases():
    total_data = cases_data.groupby('Country').agg({'Daily_Deaths': 'sum', 'Daily_New_Cases': 'sum'}).reset_index()
    descriptive_stats_cases = total_data['Daily_New_Cases'].describe()
    st.write("Descriptive Analysis of Total Confirmed Cases:")
    st.write(descriptive_stats_cases)
    
# # 5 ------------------------- Case by Country ---------------------------------------------
@profile
def cases_by_country_chart():
    cases_by_country_data = cases_data.groupby('Country')['Daily_New_Cases'].sum().reset_index()
    fig = px.choropleth(cases_by_country_data, locations='Country', locationmode='country names', 
                        color='Daily_New_Cases', hover_name='Country', 
                        title='Number of Cases by Country',
                        color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig)

@profile
def descriptive_analysis_cases_by_country():
    cases_by_country_data = cases_data.groupby('Country')['Daily_New_Cases'].sum().reset_index()
    
    # Descriptive statistics for daily new cases
    descriptive_stats_cases_by_country = cases_by_country_data['Daily_New_Cases'].describe()
    st.write("Descriptive Analysis of Daily New Cases by Country:")
    st.write(descriptive_stats_cases_by_country)
    
# # 6 ------------------------- Vaccine by country ---------------------------------------------
@profile    
def vaccine_by_country_chart():
    vaccine_by_country_data = vaccine_data.groupby('country')['vaccines'].first().reset_index()
    fig = px.bar(vaccine_by_country_data, x='country', y='vaccines', title='Vaccine Types by Country', 
                 labels={'country': 'Country', 'vaccines': 'Vaccine Types'})
    st.plotly_chart(fig)

# # 7 ------------------------- Effectiveness of country ---------------------------------------------
@profile
def effectiveness_of_vaccine_chart():
    # Group by country and calculate the death rate and most used vaccine
    death_rate_data = date_filter.groupby('Country')['Deaths'].sum().reset_index()
    vaccine_use_data = vaccine_data.groupby('country')['vaccines'].agg(lambda x: x.value_counts().idxmax()).reset_index()
    
    # Merge the two datasets
    effectiveness_data = pd.merge(death_rate_data, vaccine_use_data, left_on='Country', right_on='country', how='inner')
    effectiveness_data = effectiveness_data.rename(columns={'Deaths': 'Total Deaths', 'vaccines': 'Most Used Vaccine'})

    # Calculate total cases by country
    total_cases_by_country = date_filter.groupby('Country')['Daily_New_Cases'].sum().reset_index().rename(columns={'Daily_New_Cases': 'Total Cases'})

    # Merge total cases with effectiveness data
    effectiveness_data = pd.merge(effectiveness_data, total_cases_by_country, on='Country', how='inner')

    # Calculate death rate
    effectiveness_data['Death Rate'] = effectiveness_data['Total Deaths'] / effectiveness_data['Total Cases'] * 100

    # Plot the data
    fig = px.scatter(effectiveness_data, x='Most Used Vaccine', y='Death Rate', color='Country', title='Effectiveness of Vaccines by Death Rate')
    st.plotly_chart(fig)
    
    # Calculate the average death rate for each vaccine
    avg_death_rate_by_vaccine = effectiveness_data.groupby('Most Used Vaccine')['Death Rate'].mean().reset_index()
    
    # Find the most and least effective vaccines
    most_effective_vaccine = avg_death_rate_by_vaccine.loc[avg_death_rate_by_vaccine['Death Rate'].idxmin()]
    least_effective_vaccine = avg_death_rate_by_vaccine.loc[avg_death_rate_by_vaccine['Death Rate'].idxmax()]
    
    # Display the results
    st.write("Most effective vaccine is:", most_effective_vaccine['Most Used Vaccine'], "with an average death rate of", most_effective_vaccine['Death Rate'])
    st.write("Least effective vaccine is:", least_effective_vaccine['Most Used Vaccine'], "with an average death rate of", least_effective_vaccine['Death Rate'])


# Define main content based on sidebar choice
if menu_choice == "Home":
    # Main title and welcome message
    st.title("Covid-19 Analytics Dashboard")
    st.write("Welcome to the Covid-19 Analytics Dashboard. Use the filters in the sidebar to explore the data.")
    
    st.markdown(" ### Death Rate Chart")
    death_rate_chart()
    
    # Create columns for layout
    # Two chart align in a same row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Weekly Confirmed Case")
        weekly_confirmed_chart()
    with col2:
        st.markdown("Weekly Recovered Case")
        weekly_recovered_chart()

    st.markdown(" ### Vaccine by Country")
    vaccine_by_country_chart()
    
    st.markdown(" ### Effectiveness of Vaccines")
    effectiveness_of_vaccine_chart()
        
elif menu_choice == "Death Rate Charts":
    st.title("Death Rate Charts")
    death_rate_chart()
    
    # Define column for charts
    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Actual Death Rate Chart")
        actual_death_rate_chart()
    with colB:
        st.markdown("### Predicted Death Rate Chart")
        predicted_death_rate_chart()
    
    #st.markdown("### Descriptive Analysis of Death Rate")
    descriptive_analysis_chart()
    
elif menu_choice == "Weekly Confirmed Case":
    st.title("Weekly Confirmed Case")
    weekly_confirmed_chart()
    
    # Define column for charts
    colC, colD = st.columns(2)
    with colC:
        st.markdown("### Actual Weekly Confirmed Case")
        actual_weekly_confirmed_chart_2020()
    with colD:
        st.markdown("### Predicted Weekly Confirmed Case")
        predict_weekly_confirmed()
    descriptive_analysis_confirmed()
    
elif menu_choice == "Weekly Recovered Case":
    st.title("Weekly Recovered Case")
    weekly_recovered_chart()
    
    # Define column
    colE, colF = st.columns(2)
    with colE:
        st.markdown("Actual Weekly Recovered Case")
        actual_weekly_recovered_chart_2020()
    with colF:
        st.markdown("Predicted Weekly Recovered Case")
        predict_weekly_recovered()
    descriptive_analysis_recovered()

elif menu_choice == "Cases by Country":
    st.title("Cases by Country Chart")
    cases_by_country_chart()
    descriptive_analysis_cases_by_country()
    
elif menu_choice == "Total Death VS Cases":
    st.title("Total Death VS Cases Chart")
    total_deaths_vs_cases_chart(cases_data)
       
    # Define column
    colG, colH = st.columns(2)
    with colG:
        #st.markdown("Actual Weekly Recovered Case")
        descriptive_analysis_deaths()
    with colH:
        #st.markdown("Predicted Weekly Recovered Case")
        descriptive_analysis_cases()
     
elif menu_choice == "Vaccine by Country":
    st.title("Vaccine by Country Chart")
    vaccine_by_country_chart()
    
elif menu_choice == "Effectiveness of Vaccines":
    st.title("Effectiveness of Vaccines")
    effectiveness_of_vaccine_chart()

# Footer
footer = """<style>
a:hover, a:active { color: red; background-color: transparent; text-decoration: underline; }
.footer { position: fixed; left: 0; height:5%; bottom: 0; width: 100%; background-color: #243946; color: white; text-align: center; }
</style>
<div class="footer">
<p>From Ng Jing Ying, Ong Yu Chin, Tan Guan Yi, and Yew Zheng Hong <a style='display: block; text-align: center;' </a></p>
</div>"""
st.markdown(footer, unsafe_allow_html=True)
