# Import Python Libraries

import pandas as pd #Used for data manipulation and numerical calculations
import numpy as np  #Used for data manipulation and numerical calculations
import matplotlib.pyplot as plt #Used for data visualizations
import seaborn as sns #Used for data visualizations
#to ignore warnings
import warnings
warnings.filterwarnings('ignore') 

# Reading Dataset
vaccine_data = pd.read_csv("country_vaccinations.csv")
aggregated_COVID19_data = pd.read_csv("countries_aggregated.csv")

##############################################################################################################
# Analyzing data

#Display number of observations(rows) and features(columns) in the dataset
print("Number of observations and features for vaccine dataset:\n", vaccine_data.shape, "\n")     
print("Number of observations and features for aggregated COVID-19 dataset:\n", aggregated_COVID19_data.shape, "\n") 

# Set option to display all columns
pd.set_option('display.max_columns', None)

#Display top 5 observations of the dataset
print("Top 5 observation of vaccine dataset:\n", vaccine_data.head(), "\n")   
print("Top 5 observation of aggregated COVID-19 dataset:\n", aggregated_COVID19_data.head(), "\n")    

#Display last 5 observations of the dataset    
print("Last 5 observation of vaccine dataset:\n", vaccine_data.tail(), "\n")      
print("Last 5 observation of aggregated COVID-19 dataset:\n", aggregated_COVID19_data.tail(), "\n")  

#Unique value
print("Unique value of vaccine dataset:\n", vaccine_data.nunique(), "\n")    
print("Unique value of aggregated COVID-19 dataset:\n", aggregated_COVID19_data.nunique(), "\n")

#Understand the data type and information about the data
vaccine_data.info()
print("\n")
aggregated_COVID19_data.info()
print("\n")

##############################################################################################################
# Data cleaning

# Check for missing values
print("Missing values of vaccine dataset:\n", vaccine_data.isnull().sum(), "\n")
print("Missing values of aggregated COVID-19 dataset:\n", aggregated_COVID19_data.isnull().sum(), "\n")

# Handle missing values for vaccine dataset
# Fill the missing value of iso code with UNKNOWN
vaccine_data['iso_code'].fillna('UNKNOWN', inplace=True)

# Forward fill for numerical columns
num_cols = [
    'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
    'daily_vaccinations_raw', 'daily_vaccinations', 'total_vaccinations_per_hundred',
    'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred',
    'daily_vaccinations_per_million'
]
vaccine_data[num_cols] = vaccine_data[num_cols].fillna(0)

# Verify the missing values have been handled
print("Missing values of vaccine dataset after data cleaning:\n", vaccine_data.isnull().sum(), "\n")

# Convert 'Date' and 'date' columns to datetime format
aggregated_COVID19_data['Date'] = pd.to_datetime(aggregated_COVID19_data['Date'], format='%d/%m/%Y')
vaccine_data['date'] = pd.to_datetime(vaccine_data['date'], format='%d/%m/%Y')

# Print info of dataset
vaccine_data.info()
print("\n")
aggregated_COVID19_data.info()
print("\n")

##############################################################################################################
# Data reduction
# Removing less relevant columns
print("Default columns of vaccine dataset:\n")
country_vaccine.info()

country_vaccine = country_vaccine.drop(['iso_code', 'source_name', 'source_website'], axis=1)

# Display the remaining columns to verify
print("\nRemaining columns of vaccine dataset:\n")
country_vaccine.info()

# Create new features(columns)
# Calculate daily new cases, recovery cases and deaths
country_aggregated['Daily_New_Cases'] = country_aggregated.groupby('Country')['Confirmed'].diff().fillna(0).astype(int)
country_aggregated['Daily_Recovery_Cases'] = country_aggregated.groupby('Country')['Recovered'].diff().fillna(0).astype(int)
country_aggregated['Daily_Deaths'] = country_aggregated.groupby('Country')['Deaths'].diff().fillna(0).astype(int)

print(country_aggregated.tail())
