# solarproject
Author: Axel Nordin Fürdös - December 2023

This is my small past-time project of mine for creating domestic solar power forecasts.
The forecast is made by a machine learning model (Random Forest Regression) that uses publicly available metereology data from open-meteo.com.
The repository contains several scripts which lets you train your own model and then create new forecasts specific for you location:

  Build_Solar_RF_model.py - Creates a model for your specific photovoltaic installation (Must be run first)
  
  Make_Prediction.py - Creates a forecast for your specific photovoltaic installation (Can be run after you have created your model)
  
  Forecast_Follow_Up.py - Compares previously made forecasts to measurements to evaluate the quality of the forecasts 

To create a model you will need historical data for your photovoltaic installation (power output over time). 
If you do not have this data but still want to try this project out, you can grab placeholder data from PVGIS: https://re.jrc.ec.europa.eu/pvg_tools/en/

(In the default examples data from PVGIS is used.)

Enjoy and have fun!
