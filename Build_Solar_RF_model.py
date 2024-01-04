# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:19:22 2024

@author: Axel Nordin Fürdös
This script uses location (lat, lon) and historical PV power data as a timeseries and will build a machine learning model to predict future PV power production based of weather data.
Open-meteo.com is used for weather data (both historical and forecasts)
The model is a Random Forest Regression using SKLearn. 
The model and all training data is then saved, generating a large file >100 Mb
The script does take some minutes to run, depending on your computer and model settings.

Some information about the modeling approach:
It is assumed that the power output of the PV installation is somehow directly dependent on weather data and only weather data and that the PV power output is not timeseries dependent.
The final accuracy of the prediction is therefor not only dependent on the accuracy of the model, but also the accuracy of the weather forecast that is used.
Random Forest Regression is used as this a relatively simple approach whis usually is able to generate a decently accurate model.
A Random Forest Quantile Regression is also used to generate prediction intervals.
Finally a linear regression model is also created as a reference for performance. 

A special approach used here is that we only fit the model for datapoints where we know that the sun is up.
This is since we can now for certain that when the sun is not up, we have no power production, and we can easily predict when the sun is up by calculating sunrise and sunset.
"""
print("Starting script Build Solar RF Model...")
print(" ")
#%% 0. Imports 
# Import all required packages
# If you are using Anaconda most of these packages should be pre-installed.
# For the packages not pre-installed, you can install them by running the lines below:
# pip install openmeteo-requests
# pip install requests-cache retry-requests numpy pandas
# pip install quantile-forest

# Data import from open-meteo.com
import openmeteo_requests               
import requests_cache
from retry_requests import retry

# Data management and visualization
import pandas as pd
import numpy as np
import time
import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning and evaluation
import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, max_error
from quantile_forest import RandomForestQuantileRegressor   # Used to make prediction intervals source: https://github.com/zillow/quantile-forest

# Print version info
print("Version info:")
print('pandas: %s' % pd.__version__)
print('numpy: %s' % np.__version__)
print('sklearn: %s' % sklearn.__version__)
print(" ")
#%% 1. User Inputs
# These inputs may be changed by the user

# Input data
lat = 58.58                                                 # Location of your photovoltaic installation, Latitude  [Float]
lon = 16.16                                                 # Location of your photovoltaic installation, Longitude [Float]
tmz = "Europe/Berlin"                                       # Location of your photovoltaic installation, Timezone, see open-meteo.com for more information on format.
solar_data_file_name = 'Example_solar_data2.csv'            # CSV-file with solar data in kW. Date format must be ISO 8601 (e.g. 2022-12-31 23:00) and start-time oriented.

# Model settings
min_variable_corr = 0.25                                    # All variables with a correlation less than this will be droped.
training_test_size = 0.2                                    # How much of our input data will be used for testing
training_random_state = 22                                  # Random seed for splitting of training data.                   
training_iterations = 1                                     # Number of iterations for finding optimal model parameters. IMPORTANT: Increases runtime significantly!
training_jobs = -1                                          # How many threads can be used for finding optimal model parameters. IMPORTANT: Determines computer load and runtime. (-1 = Use all threads)
training_cv = 10                                            # Determines Cross-validation splitting strategy (Number of folds) 
                                                            # Note to self: This determines how it will use the training data in order to train the model.
                                                            # A good explaination can be found here: https://machinelearningmastery.com/k-fold-cross-validation/
                                                            # The higher this number is, the longer training would usually take.
                                                            # 10 is a common default choice.


#%% 2. Read File and Fetch Historical Data from Open-Meteo
# Fetch historical weather data from the API and add it to a dataframe

# Function for parsing dates for solar data
def parse(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M')

# Read the csv input file and get start- and end dates
solar_data = pd.read_csv(solar_data_file_name, parse_dates = ['date'], index_col=0, date_parser=parse) # Create dataframe with dates as index
sdate_str = str(solar_data.index.tolist()[0])[0:10]         # First date in file as string (yyyy-mm-dd)
edate_str = str(solar_data.index.tolist()[-1])[0:10]        # Last date in file as string (yyyy-mm-dd)

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": lat,
	"longitude": lon,
	"start_date": sdate_str,
	"end_date": edate_str,
	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "rain", "snowfall", "snow_depth", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "is_day", "shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "terrestrial_radiation", "shortwave_radiation_instant", "direct_radiation_instant", "diffuse_radiation_instant", "direct_normal_irradiance_instant", "terrestrial_radiation_instant"],
	"timezone": tmz
}
responses = openmeteo.weather_api(url, params=params)

# Process & print request info.
response = responses[0]
print("Request info:")
print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
print("From: " + sdate_str + " To: " + edate_str)
print(" ")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()                                  # API Response
hourly_data = {"date": pd.date_range(                       # Dictionary that we add the API respnse to
	start = pd.to_datetime(hourly.Time(), unit = "s"),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

# We iterate through the variables and add the API response data to our dictionary
print("Adding variables to dataframe...")
index_variable = 0
for variable in params['hourly']:
    hourly_data[variable] = hourly.Variables(index_variable).ValuesAsNumpy()    # Add the variable to dataframe
    print("Added " + variable)
    index_variable += 1                                                         # Increment counter
    
# Create dataframe
weather_data = pd.DataFrame(data = hourly_data)    
weather_data = weather_data.set_index('date')    # Set index to be dates

#%% 3. Data manipulation & encoding
# Create a new dataframe to hold all data, also encode new features.
# We only want to train the model on datapoints where the sun is up, so we keep track of these with a special index.

# Create a main dataframe that holds weather and solar data.
main_df = weather_data.copy()                           # Start with a copy of the weather data
main_df['solar power'] = solar_data['Power (kW)']       # Add the solar data
main_df = main_df.dropna()                              # Drop any NaN rows

# Create new columns for month, day, hour and sin/cos of month and hour
print("Encoding data features and removing unnecessary rows...")
main_df[['month', 'day', 'hour', 'sine month', 'cos month', 'sine hour', 'cos hour']] = 0

rows_tracker = 0                    # To keep track of how many rows we have processed so far
for index, row in main_df.iterrows():
    main_df.loc[index,'month'] = index.month
    main_df.loc[index,'day'] = index.day
    main_df.loc[index,'hour'] = index.hour
    main_df.loc[index,'sine month'] = np.sin((index.month - 1)*np.pi/11)
    main_df.loc[index,'cos month'] = np.cos((index.month - 1)*np.pi/11)
    main_df.loc[index,'sine hour'] = np.sin((index.month - 1)*np.pi/23)
    main_df.loc[index,'cos hour'] = np.cos((index.month - 1)*np.pi/23)
    rows_tracker += 1
    if(np.mod(rows_tracker,500) == 0):    
        print("Encoded " + str(rows_tracker) + "/" + str(main_df.shape[0]) + " rows")

# Index to keep track of which hours the sun is up for.        
Sun_index = main_df[main_df['is_day'] == 1].index        
print("Done encoding!")

# Create a new dataframe with data only for when the sun is up. 
sun_is_up_data = main_df.copy()                                 # Create copy
sun_is_up_data = sun_is_up_data.loc[Sun_index]                  # Only include rows when the sun is up
sun_is_up_data = sun_is_up_data.drop(columns = ['is_day'])      # Drop column as its not needed anymore
print("Dataset reduced to " + str(sun_is_up_data.shape[0]) + " rows")
print(" ")
#%% 4. Statistical Analysis
# Calculate correlation to our target variable (solar power)
# The assumption is that variables with low correlation are not needed in the model.

print("Performing correlation analysis...")
df_corr = sun_is_up_data.corr()
sorted_correlation = abs(df_corr.loc['solar power']).sort_values(ascending = False)
print("Most important features are:")
print("Feature                         Correlation")
print(sorted_correlation[2:])
print("Removing features with correlation less than " + str(min_variable_corr))
print(" ")
# Filter out columns with too low correlations
I = sorted_correlation > min_variable_corr # Remove columns with to low correlations
I = I.loc[I]
relevant_columns = I.index.tolist()        # Create a list of all relevant variables


#%% 5. Train Models
# Split our input data into a set for training data and a set for testing data.
# Train a linear regression model that will be used as a reference
# Then train the random forest model

# Split data into training set and testing set
print("Spliting dataframes into training and testings data...")
input_feat = list(set(relevant_columns).difference(set(['solar power'])))                                                                                                                                  # Which variables we will train our model on.
X_train, X_test, y_train, y_test = train_test_split(sun_is_up_data[input_feat].reset_index(drop=True) , sun_is_up_data['solar power'], test_size=training_test_size, random_state=training_random_state)   # Split data into training and testing sets
print("Done splitting data!")
print(" ")

print("Training linear model...")
lin_model = LinearRegression().fit(X_train, y_train)
print("Done training linear model!")
print(" ")

# Random Forest Regression Model
# 1. First we create a list of the hyper-parameters that we want to tune for our RF. We structre these in a dictionary "rf_grid".
# 2. Next we create our base model.
# 3. Then we create our "random" model which will be used to test out different hyper-parameter settings.
# 4. Now we train our model in order to find the best hyper-parameters.
# 5. Once we have found our hyper-parameters, we train a final model "rf_model" which will be our final model.

# Create list of each hyper-param to tune
n_estimators_list = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]    # Number of trees in random forest 
max_features_list = ['auto', 'sqrt']                                                     # Number of features to consider at every split
max_depth_list = [int(x) for x in np.linspace(10, 110, num=11)]                          # Maximum number of levels in tree
max_depth_list.append(None)                                                              # Also add the option to have no levels
min_samples_split_list = [2, 5, 10]                                                      # Mininum number of sampls required to split a node
min_samples_leaf_list = [1, 2, 4]                                                        # Minimum number of sampls required at each leaf node
bootstrap_list = [True]                                                                  # Method of selecing samples for training each tree (Always use bootstrap)

# Structure model hyper-params as a dictionary
rf_grid = {'n_estimators': n_estimators_list,
           'max_features': max_features_list,
           'max_depth': max_depth_list,
           'min_samples_split': min_samples_split_list,
           'min_samples_leaf': min_samples_leaf_list,
           'bootstrap': bootstrap_list}

# Create base LGBM model
rf_base = RandomForestRegressor(random_state=training_random_state)

# Create random search for LGBM model
rf_random = RandomizedSearchCV(estimator=rf_base, param_distributions=rf_grid, 
                                 n_iter=training_iterations, cv=training_cv, verbose=1, random_state=training_random_state, 
                                 n_jobs=training_jobs)

start_time = time.time()

print('Searching for optimal hyper parameters for Random Forest Model...')
# Fit the random search LGBM model
rf_random.fit(X_train, y_train)

print('Done searching after ' + str(round(time.time() - start_time,1)) + ' seconds')
print("Best parameters were:")
print(rf_random.best_params_)
print(" ")

print("Fitting Random Forest model...")
# Train the model with the best hyper parameters
rf_model = RandomForestRegressor(**rf_random.best_params_, random_state=training_random_state)
rf_model.fit(X_train, y_train)

# Calculate and present which features were the most important
feat_imp_score = (rf_model.feature_importances_/max(rf_model.feature_importances_)*100).tolist()
feature_ranking_with_score = dict(sorted(zip(feat_imp_score, input_feat), reverse=True))
feature_score_df = pd.DataFrame(feature_ranking_with_score.items())
feature_score_df = feature_score_df.rename(columns = {0 : 'Score', 1: 'Variable'})
print("Most important features were:")
print(feature_score_df)
print(" ")

print("Done fitting model!")
print(" ")

# Random Forest Quantile Model
# The random forest quantile model lets us create prediction intervals instead of single point predictions.
# The qunatile model is not as advanced and only uses default settings for its parameters.

print("Fitting Quantile model...")
rfqr_model = RandomForestQuantileRegressor()
rfqr_model.fit(X_train, y_train)
print("Done fitting!")
print(" ")

#%% 6. Evaluate Models
# Evaulate the performance of our models on the test data.

# Use a custom function for Mean Absolute Percentage Error
# MAPE = Average Absolute Error / Average Absolute Value
def custom_MAPE(y_true, y_pred):
    average_absolute_error = np.mean(abs(y_true - y_pred))
    average_value = np.mean(abs(y_true))
    return average_absolute_error/average_value

print("Recording predictions and performance...")
# First make and record the predictions
df_predictions = pd.DataFrame(columns = ['Test', 'Linear', 'Random Forest', 'Lower Bound (10%)', 'Upper Bound (90%)'])        # Dataframe to hold test values and predicted values
df_predictions['Test'] = y_test.ravel()                                    # Test values
df_predictions['Linear'] = lin_model.predict(X_test)                       # Predictions for linear model   
df_predictions['Random Forest'] = rf_model.predict(X_test)                 # Predictions for Random Forest model
df_predictions[['Lower Bound (10%)', 'Upper Bound (90%)']] = rfqr_model.predict(X_test, [0.10, 0.90]) # Prediction interval for Random Forest Quantile Model

# Next calculate and record the performance
df_model_performance = pd.DataFrame(index = ['R2','RMSE', 'Explained Variance', 'Mean Absolute Error', 'Mean Absolute Percentage Error', 'Maximum Error'], columns = ['Linear','Random Forest'])

for column in df_model_performance.columns.tolist():
    df_model_performance.loc['R2',column] = r2_score(df_predictions['Test'],df_predictions[column])
    df_model_performance.loc['Explained Variance',column] = explained_variance_score(df_predictions['Test'],df_predictions[column])
    df_model_performance.loc['Mean Absolute Error',column] = mean_absolute_error(df_predictions['Test'],df_predictions[column])
    df_model_performance.loc['RMSE',column] = np.sqrt(mean_squared_error(df_predictions['Test'],df_predictions[column]))
    df_model_performance.loc['Mean Absolute Percentage Error',column] = custom_MAPE(df_predictions['Test'],df_predictions[column])
    df_model_performance.loc['Maximum Error',column] = max_error(df_predictions['Test'],df_predictions[column])
  
# Add performance metric for bounds
df_model_performance['Random Forest Quantile'] = np.nan    
df_model_performance.loc['Time within bounds (%)'] = np.nan
within_bounds = (df_predictions['Test'] > df_predictions['Lower Bound (10%)']) & (df_predictions['Test'] < df_predictions['Upper Bound (90%)'])  
df_model_performance.loc['Time within bounds (%)', 'Random Forest Quantile'] = 100*np.sum(np.ones(df_predictions.shape[0])[within_bounds])/np.sum(np.ones(df_predictions.shape[0]))

# Create and calculate dataframe for prediction errors
df_prediction_errors = pd.DataFrame(index = df_predictions.index, columns = ['Linear', 'Random Forest'])
df_prediction_errors['Linear'] = (df_predictions['Linear'] - df_predictions['Test'])
df_prediction_errors['Random Forest'] = (df_predictions['Random Forest'] - df_predictions['Test'])

# Plot distributions of errors
xlim = np.max(df_predictions['Test'])

# Creating a customized histogram with a density plot for Random Forest Model
sns.histplot(df_prediction_errors['Random Forest'] , binwidth = 0.5, binrange = (-xlim,xlim), kde=True, color='orange', edgecolor='red')
plt.xlabel('Prediction Error [kW]')
plt.ylabel('Density')
plt.title('Distribution of Residuals Random Forest Model')
plt.show()

print("Performance:")
print(" ")
print(df_model_performance)
print(" ")


#%% 7. Save Results
# We create a dictionary "results" that we add all our interresting results to which we then can save to a file.

datecreated = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") # When the file was created
filename = "Solar RF Model_" + datecreated + "_.pkl"
filename = filename.replace(":","-") # Replace ":" as we cant have this symbol in filenames...

results = {'Filename' : filename,
           'Date created' : datecreated,
           'Original Data' : main_df,
           'Longitude' : lon,
           'Latitude' : lat,
           'Minimum Correlation' : min_variable_corr,
           'Input features': input_feat,
           'API Request Params': params,
           'Linear Model' : lin_model,
           'Random Forest Model' : rf_model,
           'Random Forest Quantile Model' : rfqr_model,
           'Feature Ranking' : feature_ranking_with_score,
           'X train' :  X_train,
           'X test' : X_test,
           'y train' : y_train,
           'y test' : y_test
    }

print("Saving data to file: " + filename)
# Open the file
file = open(filename, 'wb')
# Dump to file
pickle.dump(results,file)
# Close the file
file.close()
print("Done saving!")

