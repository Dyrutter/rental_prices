import argparse
import logging
import os
from distutils.util import strtobool
import pandas as pd
import numpy as np
import wandb
from sklearn.impute import SimpleImputer


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
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


def drop_features(df):
    """
    Features chosen for dropping from Pandas profile
    Drop 'availablility_365' because 35.9% missing values
    Drop 'id,' 'host_id,' 'host_name', and 'neighbourhood' for cardinality
    Drop 'number_of_reviews' and 'reviews_per_month' b/c of high correlation
    """
    df = df.drop(['id', 'host_id', 'host_name', 'neighbourhood',
                  'reviews_per_month'], axis=1)
    return df


def drop_useless(df):
    """
    Drop duplicates
    Drop any samples missing the 'price' input
    """
    df = df.drop_duplicates().reset_index(drop=True)
    df.dropna(subset=['price'], inplace=True)
    return df


def drop_outliers(df):
    """
    Drop extreme outliers for 'minimum_nights', 'last_review', and
        'calculated_host_listings_count'
        (found in pandas profiling report)
    Drop all latitudes and longitudes not in NYC
    """
    # Drop major outliers from minimum nights
    idx = df['minimum_nights'].between(args.min_nights, args.max_nights)
    df = df[idx].copy()

    # Drop major outliers from last review
    idx = df['last_review'].between(0, 50)
    df = df[idx].copy()

    # Drop major outliers from calculated_host_listings_count
    idx = df['calculated_host_listings_count'].between(1, 5)
    df = df[idx].copy()

    # Drop lats and longs not in NYC
    idx = df['longitude'].between(-74.25, -73.50) &\
        df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Drop price outliers
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    return df


def go(args):

    # Instantiate wandb run and get raw data artifact
    run = wandb.init(job_type="process_data")
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact, type='raw_data')
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path, low_memory=False)

    # Change dates column
    logger.info("Reworking dates")
    df = engineer_dates(df)

    # Drop unused features, duplicates, and outliers
    logger.info("Dropping unused features")
    df = drop_features(df)
    logger.info("Dropping duplicates and rows missing 'price'")
    df = drop_useless(df)
    logger.info("Dropping outliers")
    df = drop_outliers(df)

    filename = args.output_name  # "preprocessed_data.csv"

    # Save clean df to local machine if desired
    if args.save_locally:
        df2 = df.copy()
        local = 'finaldata.csv'
        df2.to_csv(os.path.join(os.getcwd(), local))
    df.to_csv(args.output_name, index=False)

    # Create artifact and upload to wandb
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

    parser.add_argument(
        "--min_nights",
        type=float,
        help="minimum nights lower param",
        required=True
    )

    parser.add_argument(
        "--max_nights",
        type=float,
        help="minimum nights upper param",
        required=True
    )

    # True values are y, yes, t, true, on and 1;
    # False values are n, no, f, false, off and 0
    # Will raise ValueError if input argument is not of proper type
    parser.add_argument(
        "--save_locally",
        type=lambda x: bool(strtobool(x)),
        help='Choose whether or not to save clean data frame to local file',
        required=True
    )

    args = parser.parse_args()

    go(args)
