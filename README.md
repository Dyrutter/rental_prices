# PROJECT OVERVIEW

This is a demonstration of MLOps using MLFLow and Weights & Biases
In this example, a property management company is renting rooms and properties for short periods of time on various platforms. We estimate the typical price for a given property based on the price of similar properties. New data is received in bulk on a frequent basis. With each addition, the model is retrained with the same cadence, necessitating an end-to-end pipeline in production.
The pipeline has 7 components:
+ Download data
+ Clean data
+ A data integrity check using Pytest
+ An EDA analysis, which creates an ipython notebook
+ Data Segregation (Splitting into train and test sets)
+ Create a model
+ Test the model
Each component can be run independently, assuming its previous component(s) have been run at least once.
The integrity of each upload is confirmed through GitHub Actions via Pytest and flake8

# ENVIRONMENT SETUP

## PREREQUISITES

+ Weights and Biases account, which can be created [here](https://wandb.ai/site)
+ GitHub account (for running)
+ Clone GitHub repo `https://github.com/Dyrutter/rental_prices.git`

## DEPENDENCIES
+ Install requirements found in [requirements](./requirements.txt)
+ A python 3.10 `conda` virtual environment
+ Note: This project was created using MacOS Monterey

# PRIMARY FILES

## [main.py (Root Directory)](./main.py)
+ Defines each MLFLow component
+ Specifies hyperparameters for each component using argparse in conjunction with [Hydra](https://hydra.cc/docs/intro/)
+ Specifies input and output Weights & Biases artifacts
+ From root directory, can be run locally with command `mlflow run .`
+ Specific components (e.g. "download") can be run locally with `mlflow run . -P hydra_options="main.execute_steps='download'"`
+ Can be run on GitHub using command `mlflow run https://github.com/DyRutter/rental_prices.git -v 1.0.2` 
+ A new data sample (new_sample.csv) can be input using `mlflow run https://github.com/DyRutter/rental_prices.git -v 1.0.2 -P
    hydra_options="data.sample='new_sample.csv'"`
    
## [download_data.py (download component)](./download/download_data.py)
+ Gets data from url specified in [hydra config file](./config/config.yaml)
+ Converts data into a Weights & Biases artifact
+ Uploads the artifact to Weights & Biases

## [run.py (basic cleaning component)](./basic_cleaning/run.py)
+ Gets raw data artifact (created in download component) from Weights & Biases
+ Drops duplicates
+ Drops price outliers according to min and max specified in [hydra config file](./config/config.yaml)
+ Converts dates to datetime format using Pandas
+ Creates cleaned data csv file 
+ Converts cleaned data file into an artifact and uploads it to Weights and Biases

## [test_data.py (check data component)](./check_data/test_data.py)
+ In conjunction with [conftest.py](./check_data/conftest.py), asserts data integrity using Pytest
+ Confirms column names, neighborhood names, and row counts are as expected
+ Asserts latitude and longitude boundaries are within the expected values for NYC
+ Confirms price ranges are between the min and max values specified in [hydra config file](./config/config.yaml)
    
## [EDA.ipynb (EDA component)](./EDA/EDA.ipynb)
+ Downloads raw data artifact from Weights and Biases
+ Runs y data profiling (formerly pandas_profiling) analysis

## [run.py (segregate component)](./segregate/run.py)
+ Splits data into train and test sets
+ Uploads data sets as artifacts to Weights and Biases

## [run.py (random_forest component)](./random_forest/run.py)
+ Fits data to a random forest inference pipeline
+ Creates a scikit-learn chart of feature importances
+ Uploads model as an artifact to Weights and Biases

## [run.py (test_regression_model component)](./test_regression_model/run.py)
+ Retrieves model artifact from Weights and Biases
+ Calculates R squared score and mean absolute error
+ Uploads scores to Weights and Biases

## OTHER FILES
+ A conda.yaml dependencies file exists in each component for use by MLFlow
+ An MLproject configuration file exists in each component for use by MLFLow
