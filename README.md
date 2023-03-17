# Project Overview
This is a demonstration of MLOps using MLFLow and Weights & Biases
In this example, a property management company is renting rooms and properties for short periods of time on various platforms. We estimate the typical price for a given property based on the price of similar properties. New data is received in bulk on a frequent basis. With each addition, the model is retrained with the same cadence, necessitating an end-to-end pipeline in production.
The pipeline has 7 components:
+ Download data
+ Clean data
+ A data integrity check using pytest
+ An EDA analysis, which creates an ipython notebook
+ Data Segregation (Splitting into train and test sets)
+ Create a model
+ Test the model
Each component can be run independently, assuming its previous component(s) have been run at least once. 
