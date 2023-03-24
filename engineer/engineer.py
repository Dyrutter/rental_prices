import argparse
import tempfile
import itertools
import setuptools
import yaml
import logging
import os
import shutil
import matplotlib.pyplot as plt
import wandb
import mlflow
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def process_name(df):
    """
    Apply TfidfVectorizer to 'name' column
    For 'host_name' and 'neighbourhood' use Count Vectorizer, b/c looking at
    	n-grams is the most effective way to analyze single names
    """
    # Instantiate vectorizer
    name_tfidf = TfidfVectorizer(
            binary=False,
            max_features=10,
            stop_words=None,
            token_pattern='(?u)\\b\\w+\\b')

    # Get name column as a series & transform using tf_idf
    # Returned type is scipy sparse matrix, so must use toarray()
    name_col = pd.Series(df['name'])
    name_vect = name_tfidf.fit_transform(name_col).toarray()

    # Create a data frame from tf_idf, using column_x strings for columns
    name_df = pd.DataFrame(
    	name_vect,
    	columns=['Name_' + str(i + 1)  for i in range(name_vect.shape[1])])

    # Merge data frames along columns, use reindex to keep consistent rows
    df = pd.concat([df, name_df], axis=1).reindex(df.index)
    df = df.drop(['name'], axis=1)
    return df

def process_host(df):
    """
    Apply Count Vectorizer to 'host_name', b/c looking at
    	n-grams is the most effective way to analyze single names
    """
    host_count = CountVectorizer(
    	ngram_range=(2, 3),
    	max_features=10,
    	analyzer="char")

    host_col = pd.Series(df['host_name'])
    host_vect = host_count.fit_transform(host_col).toarray()
    host_df = pd.DataFrame(
    	host_vect,
    	columns=['Host_' + str(i+1) for i in range(host_vect.shape[1])])

    df = pd.concat([df, host_df], axis=1).reindex(df.index)
    df = df.drop(['host_name'], axis=1)
    return df


def process_neigh(df):
	"""
	Apply Count Vectorizer to 'neighbourhood', b/c looking at
	n-grams is the most effective way to analyze single names
	"""
	neigh_count = CountVectorizer(
		ngram_range=(2,3),
		max_features=10,
		analyzer='char')

	neigh_col = pd.Series(df['neighbourhood'])
	neigh_vect = neigh_count.fit_transform(neigh_col).toarray()
	neigh_df = pd.DataFrame(
		neigh_vect,
		columns=['neigh_' + str(i+1) for i in range(neigh_vect.shape[1])])
	df = pd.concat([df, neigh_df], axis=1).reindex(df.index)
	df = df.drop(['neighbourhood'], axis=1)
	return df

def normal(df):
	"""
	Apply normalization to data frame
	"""
	# Fill the nas left by vectorizers
	df = df.fillna(0)
	# Separate features and labels
	labels = df['price']
	features = df.drop(['price'], axis=1)

	# Normalize features and return labels to data frame
	norm = normalize(features)
	features = pd.DataFrame(norm, columns=features.columns)
	features['price'] = labels
	return features

def go():
	pass


## REMEMBER TO DROP LABELS BEFORE NORMALIZING