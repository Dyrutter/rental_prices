import argparse
import logging
import os
import pandas as pd
import wandb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def engineer_dates(data_frame):
    """
    Convert date column to feature that represents the number of days passed
    since the last review.
    Impute missing date values from an old date, because there hasn't been a
    review for a long time
    """
    # Instantiate imputer & reshape date column for compatability
    date_imputer = SimpleImputer(strategy='constant', fill_value='2010-01-01')
    date_column = np.array(data_frame['last_review']).reshape(-1, 1)

    # Impute dates and convert to datetime
    dates_imputed = date_imputer.fit_transform(date_column)
    dates_df = pd.DataFrame(dates_imputed).apply(pd.to_datetime)

    # Convert datetime days to ints & get the distance between them
    dates_df = dates_df.apply(lambda d: (d.max() - d).dt.days, axis=0)
    data_frame['last_review'] = dates_df
    return data_frame


def encode_room_type(data_frame):
    """
    Use an ordianl encoder, because order is meaningful in room type:
    'Entire home/apt' > 'Private room' > 'Shared room'
    Note: Don't need to impute, because mandatory on website input
    """
    encoder = OrdinalEncoder()
    room_type = np.array(data_frame['room_type']).reshape(-1, 1)
    encoded = encoder.fit_transform(room_type)
    data_frame['room_type'] = encoded
    return data_frame


def encode_group(data_frame):
    """
    Encode the categorical feature 'neighbourhood_group'
    No need to impute, because it was a required input
    """
    data_frame['neighbourhood_group'] =\
        data_frame['neighbourhood_group'].astype('category')
    data_frame['neighbourhood_group'] =\
        data_frame['neighbourhood_group'].cat.codes
    return data_frame


def impute_text(data_frame):
    """
    Impute missing values in "name" and "host_name" columns
    """
    # Instantiate imputers
    name_imputer = SimpleImputer(strategy="constant", fill_value="")
    host_imputer = SimpleImputer(strategy="constant", fill_value="")
    neighbourhood_imputer = SimpleImputer(strategy="constant", fill_value="")

    # Impute "name" column
    name_column = np.array(data_frame['name']).reshape(-1, 1)
    names_imputed = name_imputer.fit_transform(name_column)
    data_frame['name'] = names_imputed

    # Impute "host_name" column
    host_column = np.array(data_frame['host_name']).reshape(-1, 1)
    hosts_imputed = host_imputer.fit_transform(host_column)
    data_frame['host_name'] = hosts_imputed

    # Impute "neighbourhood" column
    neigh_column = np.array(data_frame['neighbourhood']).reshape(-1, 1)
    neighs_imputed = neighbourhood_imputer.fit_transform(neigh_column)
    data_frame['neighbourhood'] = neighs_imputed
    return data_frame


def drop_useless(df):
    """
    Drop duplicates, price outliers, "id" & "host_id" columns, and all
    latitudes and longitudes not in NYC
    """
    # Drop duplicates and outliers
    df = df.drop_duplicates().reset_index(drop=True)
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Drop 'id' and 'host_id' columns
    df = df.drop(['id', 'host_id'], axis=1)

    # Drop lats and longs not in NYC
    idx = df['longitude'].between(-74.25, -73.50) &\
        df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Drop NA rows in ['price'], the label column
    df.dropna(subset=['price'], inplace=True)
    return df


def impute_numerics(df):
    """
    Impute missing values in numeric columns "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365", "longitude", and
    "latitude"
    """
    imputer = SimpleImputer(strategy="most_frequent")
    col = np.array(df["reviews_per_month"]).reshape(-1, 1)
    imputed_col = imputer.fit_transform(col)
    df["reviews_per_month"] = imputed_col

    imputer = SimpleImputer(strategy="most_frequent")
    col = np.array(df["minimum_nights"]).reshape(-1, 1)
    imputed_col = imputer.fit_transform(col)
    df["minimum_nights"] = imputed_col

    imputer = SimpleImputer(strategy="most_frequent")
    col = np.array(df["number_of_reviews"]).reshape(-1, 1)
    imputed_col = imputer.fit_transform(col)
    df["number_of_reviews"] = imputed_col

    imputer = SimpleImputer(strategy="most_frequent")
    col = np.array(df["calculated_host_listings_count"]).reshape(-1, 1)
    imputed_col = imputer.fit_transform(col)
    df["calculated_host_listings_count"] = imputed_col

    imputer = SimpleImputer(strategy="most_frequent")
    col = np.array(df["availability_365"]).reshape(-1, 1)
    imputed_col = imputer.fit_transform(col)
    df["availability_365"] = imputed_col

    imputer = SimpleImputer(strategy="most_frequent")
    col = np.array(df["longitude"]).reshape(-1, 1)
    imputed_col = imputer.fit_transform(col)
    df["longitude"] = imputed_col

    imputer = SimpleImputer(strategy="most_frequent")
    col = np.array(df["latitude"]).reshape(-1, 1)
    imputed_col = imputer.fit_transform(col)
    df["latitude"] = imputed_col
    return df


def go(args):

    # Initialize run and get raw data file
    run = wandb.init(job_type="process_data")
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact, type='raw_data')
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path, low_memory=False)

    # Impute missing values, encode categoricals, & engineer dates
    logger.info("Imputing missing values, encoding, and engineering dates")
    df = engineer_dates(df)
    df = encode_room_type(df)
    df = encode_group(df)
    df = impute_text(df)
    df = impute_numerics(df)

    # Drop useless features
    logger.info("Dropping duplicates, outliers, and useless features")
    df = drop_useless(df)

    filename = args.output_name  # "preprocessed_data.csv"
    df.to_csv(args.output_name, index=False)  # added index=False

    artifact = wandb.Artifact(
        name=args.output_name,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--output_name",
        type=str,
        help="Name for the artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help='minimum price',
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help='maximum price',
        required=True
    )

    args = parser.parse_args()

    go(args)
