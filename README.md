# PROJECT OVERVIEW

This is a demonstration of an MLOps pipeline using MLFLow and Weights & Biases.

In this example, a property management company is renting rooms and properties for short periods of time on various platforms. I estimate the typical price for a given property based on the price of similar properties. New data is received in bulk on a frequent basis. With each addition, the model is retrained with the same cadence, necessitating an end-to-end pipeline in production.

The pipeline has 7 components:
+ Download raw data
+ Confirm raw data's integrity using Pytest
+ An EDA analysis using pandas profiling, which creates an ipython notebook
+ Clean/preprocess data based on pandas profiling findings
+ Confirm preprocessed data's integrity using Pytest
+ Data Segregation (Splitting into train and test sets)
+ Create a model
+ Test the model's performance metrics & produce analytic images

Each component can be run independently, assuming its previous component(s) have been run at least once.
The integrity of each commit is confirmed through GitHub Actions via Pytest and flake8

# ENVIRONMENT SETUP

## PREREQUISITES

+ Weights and Biases account, which can be created [here](https://wandb.ai/site)
+ GitHub account (for running)
+ Clone GitHub repo `https://github.com/Dyrutter/rental_prices.git`
+ A python 3.10 `conda` virtual environment

## DEPENDENCIES

+ Install requirements found in [requirements.txt file](./requirements.txt)
+ Note: This project was created using MacOS Monterey

# PRIMARY FILES

### [main.py (Root Directory)](./main.py)
+ Defines each MLFLow component
+ Specifies hyperparameters for each component using argparse in conjunction with [Hydra](https://hydra.cc/docs/intro/)
+ Specifies input and output Weights & Biases artifacts
+ From root directory, can be run locally with command `mlflow run .`
+ Specific components (e.g. "download") can be run locally with `mlflow run . -P hydra_options="main.execute_steps='download'"`
+ Can be run on GitHub using command `mlflow run https://github.com/DyRutter/rental_prices.git -v 1.0.2` 
+ A new data sample (new_sample.csv) can be input using `mlflow run https://github.com/DyRutter/rental_prices.git -v 1.0.2 -P
    hydra_options="data.sample='new_sample.csv'"`
    
### [download_data.py (download component)](./download/download_data.py)
+ Gets data from url specified in [hydra config file](./config/config.yaml)
+ Converts data into a Weights & Biases artifact
+ Uploads the artifact to Weights & Biases

### [test_raw_data.py (Test Raw Data component)](./test_raw_data/test_raw.py)
+ In conjunction with [conftest.py](./test_raw_data/conftest.py), confirm the integrity of raw input data using Pytest
+ Confirms the input raw data has correct column names
+ Confirms neighbourhood names and room types are within the expected values

### [run.py (basic cleaning component)](./basic_cleaning/run.py)
+ Gets raw data artifact (created in download component) from Weights & Biases
+ Drops duplicates and useless features
+ Drops outliers according to min and max specified in [hydra config file](./config/config.yaml)
+ Engineers dates to date feature 
+ Creates cleaned data csv file and uploads it to Weights & Biases

### [test_data.py (check data component)](./check_data/test_data.py)
+ (./check_data/conftest.py), confirms preprocessing step was successful using Pytest
+ Asserts latitude and longitude boundaries are within the expected values for NYC
+ Confirms price ranges are between the min and max values specified in [hydra config file](./config/config.yaml)
+ Assesses KL divergence to confirm values aren't too dissimilar
    
### [EDA.ipynb (EDA component)](./EDA/EDA.ipynb)
+ Downloads raw data artifact from Weights and Biases
+ Runs y data profiling (formerly pandas_profiling) analysis

### [run.py (segregate component)](./segregate/run.py)
+ Splits data into train and test sets
+ Uploads data sets as artifacts to Weights and Biases

### [run.py (random_forest component)](./random_forest/run.py)
+ Fits data to a random forest inference pipeline
+ Creates a scikit-learn chart of feature importances
+ Uploads model as an artifact to Weights and Biases

### [run.py (test_regression_model component)](./test_regression_model/run.py)
+ Retrieves model artifact from Weights and Biases
+ Calculates R squared score and mean absolute error
+ Uploads scores to Weights and Biases

### [config.yaml](./config/config.yaml)
+ Primary Hydra configuration file
+ Specifies data to use and components to execute
+ Contains customization options for the random forest model

### OTHER FILES
+ A conda.yaml dependencies file exists in each component for use by MLFlow
+ An MLproject configuration file exists in each component for use by MLFLow

# ADDITIONAL RESOURCES

+ [Create Reusable ML Modules with MLflow Projects & Docker](https://towardsdatascience.com/create-reusable-ml-modules-with-mlflow-projects-docker-33cd722c93c4)
+ [MLOps-Reducing the technical debt of Machine Learning](https://medium.com/mlops-community/mlops-reducing-the-technical-debt-of-machine-learning-dac528ef39de)
+ [MLOps Core](https://ml-ops.org/content/references.html)
+ [MLOps: From a Data Scientist's Perspective](https://neptune.ai/blog/mlops)
+ [Choosing the Right Metric For Evaluating ML Models](https://www.kaggle.com/code/vipulgandhi/how-to-choose-right-metric-for-evaluating-ml-model/notebook)


# Suggestions
+ In the EDA include visualizations and other data cleaning steps. This should get even better performance from the model.
+ Explore other models beyond the RandomForest, creating a new separate step or customizing the random forest to accommodate different types of models.
+ Add discussion to a README file concerning other changes you might consider in future pipeline releases
