# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 12:58:42 2024

@author: Axel
This script will compare a forecast to its measurement
"""

#%% 0. Imports 
# Import all required packages
# If you are using Anaconda most of these packages should be pre-installed.

# Import all required packages

# Data management and visualization
import pandas as pd
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Evaluation
import sklearn
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, max_error

# Print version info
print("Version info:")
print('pandas: %s' % pd.__version__)
print('numpy: %s' % np.__version__)
print('sklearn: %s' % sklearn.__version__)
print(" ")

#%% 1. User Inputs
# These inputs may be changed by the user
# The comparasion will only be done for overlapping dates.
forecast_file_name = 'Example Solar Forecast.csv' # File name of a previously created forecast
solar_data_file_name = 'Example_solar_data3.csv'  # CSV-file with solar data in [kW]. Date format must be ISO 8601 (e.g. 2022-12-31 23:00) and start-time oriented.
PV_capacity = 15                                  # The maximum capacity of your PV installation in kW, only used for plotting.

#%% 2. Read Files

# Function for parsing dates for solar data
def parse(x):
    return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M')

# Read the solar data file
solar_data = pd.read_csv(solar_data_file_name, parse_dates = ['date'], index_col=0, date_parser=parse) # Create dataframe with dates as index

# Read the forecast file
forecast_data = pd.read_csv(forecast_file_name)
forecast_data = forecast_data.set_index('date') # Set date to index

# No create a merged dataset
main_df = forecast_data.copy()                  # Start with a copy of the forecast dataset
main_df['True'] = solar_data['Power (kW)']      # Add the measurement    
main_df['plot date'] = main_df.index.tolist()   # Also create a column with dates for plotting
main_df = main_df.dropna()                      # Remove NaN rows

# Get start and end dates from the main dataframe
sdate_str = str(main_df.index.tolist()[0])[0:10]         # First date in file as string (yyyy-mm-dd)
edate_str = str(main_df.index.tolist()[-1])[0:10]        # Last date in file as string (yyyy-mm-dd)

#%% 3. Evaluate Performance

# Use a custom function for Mean Absolute Percentage Error
# MAPE = Average Absolute Error / Average Absolute Value
def custom_MAPE(y_true, y_pred):
    average_absolute_error = np.mean(abs(y_true - y_pred))
    average_value = np.mean(abs(y_true))
    return average_absolute_error/average_value


# Next calculate and record the performance
df_model_performance = pd.DataFrame(index = ['R2','RMSE', 'Explained Variance', 'Mean Absolute Error', 'Mean Absolute Percentage Error', 'Maximum Error'], columns = ['Forecast Performance'])

for column in df_model_performance.columns.tolist():
    df_model_performance.loc['R2',column] = r2_score(main_df['True'],main_df['Main Forecast'])
    df_model_performance.loc['Explained Variance',column] = explained_variance_score(main_df['True'],main_df['Main Forecast'])
    df_model_performance.loc['Mean Absolute Error',column] = mean_absolute_error(main_df['True'],main_df['Main Forecast'])
    df_model_performance.loc['RMSE',column] = np.sqrt(mean_squared_error(main_df['True'],main_df['Main Forecast']))
    df_model_performance.loc['Mean Absolute Percentage Error',column] = custom_MAPE(main_df['True'],main_df['Main Forecast'])
    df_model_performance.loc['Maximum Error',column] = max_error(main_df['True'],main_df['Main Forecast'])
  
# Add performance metric for bounds
df_model_performance['Random Forest Quantile'] = np.nan    
df_model_performance.loc['Time within bounds (%)'] = np.nan
within_bounds = (main_df['True'] + 0.001 > main_df['Lower Bound']) & (main_df['True'] - 0.001 < main_df['Upper Bound'])         # There are issues comparing 0 < 0, so a small tolerance is added.
df_model_performance.loc['Time within bounds (%)', 'Random Forest Quantile'] = 100*np.sum(np.ones(main_df.shape[0])[within_bounds])/np.sum(np.ones(main_df.shape[0]))


print("Performance:")
print(" ")
print(df_model_performance)
print(" ")

#%% 4. Make Plot

plt.style.use('_mpl-gallery')                   # Set plot style
matplotlib.rc('font', **{'size'   : 20})        # Change font size
fig, ax = plt.subplots()                        # Create plot
fig.set_size_inches(18.5, 10.5, forward=True)   # Set figure size
fig.set_dpi(100)                                # Set resolution
ax.fill_between(main_df['plot date'], main_df['Lower Bound'], main_df['Upper Bound'], alpha=.5, linewidth=0) # Plot prediction interval
ax.plot(main_df['plot date'], main_df['Main Forecast'], linewidth=4, color='black')                          # Plot main forecast
ax.plot(main_df['plot date'], main_df['True'], linewidth=2, color='blue', linestyle = '--')                  # Plot measurement

# Calculate x-ticks and more
step_length = int(main_df.shape[0]/42)
ax.set(ylim = [0,PV_capacity*1.05], xticks = main_df['plot date'].iloc[::step_length])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.ylabel('Solar power [kW]')
plt.xticks(rotation=90)
plt.title('Solar Power Forecast from ' + sdate_str + ' to ' + edate_str)
plt.legend(['Prediction Interval (10 - 90 %)', 'Forecast', 'Measurement'])
plt.show()
