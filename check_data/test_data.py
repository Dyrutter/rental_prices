import pandas as pd
import scipy.stats
import numpy as np


def test_similar_neigh_distrib(
        data: pd.DataFrame,
        ref_data: pd.DataFrame,
        kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the
    distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for
    properties in and around NYC
    """
    idx = data['longitude'].between(
        -74.25, - 73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_price_nans_dropped(data: pd.DataFrame):
    """
    Confirm NaN values in price (label) column were dropped
    """
    assert data['price'].isna().sum() == 0
