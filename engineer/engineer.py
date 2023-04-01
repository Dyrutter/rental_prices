import argparse
import tempfile
# import itertools
# import setuptools
# import yaml
import logging
import os
from distutils.util import strtobool
# import shutil
import wandb
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer, normalize
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def process_coordinates(data_frame):
    """
    Impute missing latitude and longitude values, and confirm they
    fall within the expected range for New York
    """
    # Impute latitude values
    lat_imputer = SimpleImputer(strategy='most_frequent')
    lat_col = np.array(data_frame['latitude']).reshape(-1, 1)
    lat_imputed = lat_imputer.fit_transform(lat_col)
    data_frame['latitude'] = lat_imputed

    # Impute longitude values
    long_imputer = SimpleImputer(strategy='most_frequent')
    long_col = np.array(data_frame['longitude']).reshape(-1, 1)
    long_imputed = long_imputer.fit_transform(long_col)
    data_frame['longitude'] = long_imputed

    # Drop lats and longs not in NYC
    idx = data_frame['longitude'].between(-74.25, -73.50) &\
        data_frame['latitude'].between(40.5, 41.2)
    data_frame = data_frame[idx].copy()
    return data_frame


# check that imputation worked on NaNs
def process_dates(data_frame):
    """
    Convert date column to feature that represents the number of days passed
    since the last review.
    Impute missing date values from an old date, because there hasn't been a
    review for a long time
    """
    # Reshape date column for compatability and convert to datetime
    date_column = np.array(data_frame['last_review']).reshape(-1, 1)
    date_sanitized = pd.DataFrame(date_column).apply(pd.to_datetime)

    # Convert datetime days to ints & get the distance between them
    # d.max() is most recent day (2019-07-08) ! d is the date
    # .dt.days is applied to a date range between start and end dates
    # returning the difference between both dates in days
    dates_col = date_sanitized.apply(
        lambda d: (d.max() - d).dt.days, axis=0).to_numpy()

    # Impute dates and add imputed column to data frame
    date_imputer = SimpleImputer(strategy='median')
    dates_imputed = date_imputer.fit_transform(dates_col)
    data_frame['last_review'] = dates_imputed
    return data_frame


def process_name(df):
    """
    Impute missing values in 'name' column, then apply TfidfVectorizer
    Returns a data frame with the original columns alongside new columns
    for each tf-idf feature
    For 'host_name' and 'neighbourhood' use Count Vectorizer, b/c looking at
        n-grams is the most effective way to analyze single names
    """
    # Tf-idf accepts only a 1D array, whereas simple imputer only accepts 2D
    # feature_names_out='one-to-one' determines list of feature names
    # returned by get_feature_names_out method
    reshape_to_1d = FunctionTransformer(np.reshape,
                                        kw_args={"newshape": -1},
                                        feature_names_out='one-to-one')

    # make pipeline that imputes, reshapes, and vectorizes
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=args.tfidf_max_features,
            stop_words='english'
        ),
    )

    # Squeeze name column series to a scalar & transform using tf_idf
    # Returned type is scipy sparse matrix, so must use toarray()
    name_col = pd.Series(df['name']).squeeze().values.reshape(-1, 1)
    name_vect = name_tfidf.fit_transform(name_col).toarray()

    # Create a data frame from tf_idf, using "column_x" strings for columns
    name_df = pd.DataFrame(
        name_vect,
        columns=['Name_' + str(i + 1) for i in range(name_vect.shape[1])])

    # Merge name df with main df along columns, reindex for consistent rows
    df = pd.concat([df, name_df], axis=1).reindex(df.index)
    df = df.drop(['name'], axis=1)
    return df


def process_room_type(data_frame):
    """
    Use an ordianl encoder, because order is meaningful in room_type:
    'Entire home/apt' > 'Private room' > 'Shared room'
    Result of OrdinalEncoder is a single column of transformed features
    Function returns transformed data frame
    """
    # Instantiate encoder and imputer
    encoder = OrdinalEncoder()
    imputer = SimpleImputer(strategy="most_frequent")

    # Get room type column, impute and encode
    room_type = np.array(data_frame['room_type']).reshape(-1, 1)
    imputed = imputer.fit_transform(room_type)
    encoded = encoder.fit_transform(imputed)

    # Substitute categorical column for encoded column
    data_frame['room_type'] = encoded
    return data_frame


def process_group(data_frame):
    """
    Impute and One-hot encode the categorical feature 'neighbourhood_group'
    Returns imputed data frame with additional columns of the format:
        'neighbourhood_group_brooklyn'
    """
    # Impute missing values
    imputer = SimpleImputer(strategy="most_frequent")
    col = np.array(data_frame["neighbourhood_group"]).reshape(-1, 1)
    imputed_col = imputer.fit_transform(col)
    data_frame["neighbourhood_group"] = imputed_col

    # One Hot encode
    # Could be an issue here. Might not want to convert from sparse
    neigh = OneHotEncoder()
    encoded = neigh.fit_transform(data_frame[['neighbourhood_group']])
    col_name = neigh.get_feature_names_out(['neighbourhood_group'])

    # Returns a sparse one-hot matrix, so must convert to dense
    df = pd.DataFrame(encoded.todense(), columns=col_name)

    # Concatenate one-hot df with main df and drop categorical feature
    data_frame = pd.concat([data_frame, df], axis=1).reindex(df.index)
    data_frame = data_frame.drop(['neighbourhood_group'], axis=1)
    return data_frame


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
        columns=['Host_' + str(i + 1) for i in range(host_vect.shape[1])])

    df = pd.concat([df, host_df], axis=1).reindex(df.index)
    df = df.drop(['host_name'], axis=1)
    return df


def process_neigh(df):
    """
    Apply Count Vectorizer to 'neighbourhood', b/c looking at
    n-grams is the most effective way to analyze single names
    """
    neigh_count = CountVectorizer(
        ngram_range=(2, 3),
        max_features=10,
        analyzer='char')

    neigh_col = pd.Series(df['neighbourhood'])
    neigh_vect = neigh_count.fit_transform(neigh_col).toarray()
    neigh_df = pd.DataFrame(
        neigh_vect,
        columns=['neigh_' + str(i + 1) for i in range(neigh_vect.shape[1])])
    df = pd.concat([df, neigh_df], axis=1).reindex(df.index)
    df = df.drop(['neighbourhood'], axis=1)
    return df


def impute_numerics(df):
    """
    Impute missing values in numeric columns "minimum_nights",
    "number_of_reviews", and "availability_365"
    """
    # Impute "minimum nights" Highly-skewed, so use median
    imputer = SimpleImputer(strategy="median")
    col = np.array(df["minimum_nights"]).reshape(-1, 1)
    imputed_col = imputer.fit_transform(col)
    df["minimum_nights"] = imputed_col

    # Impute "calculated_host_listings_count"
    imputer = SimpleImputer(strategy="most_frequent")
    col = np.array(df["calculated_host_listings_count"]).reshape(-1, 1)
    imputed_col = imputer.fit_transform(col)
    df["calculated_host_listings_count"] = imputed_col

    # Impute "availability_365. 20.4% are zeroes
    imputer = SimpleImputer(strategy="median")
    col = np.array(df["availability_365"]).reshape(-1, 1)
    imputed_col = imputer.fit_transform(col)
    df["availability_365"] = imputed_col
    return df


def drop_outliers(df):
    """
    Drop extreme outliers as discovered in y_data profiling
    """
    # Drop major outliers from last review, default (0,50)
    idx = df['last_review'].between(args.min_date, args.max_date)
    df = df[idx].copy()

    # Drop outliers for minimum nights stayed, default (0,4)
    idx = df['minimum_nights'].between(args.min_nights, args.max_nights)
    df = df[idx].copy()

    # Drop major outliers from calculated_host_listings_count, default (1,5)
    idx = df['calculated_host_listings_count'].between(
        args.min_listings, args.max_listings)
    df = df[idx].copy()

    # Drop undesired price values
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
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
    # Instantiate wandb run and get train data artifact
    run = wandb.init(job_type="engineer_data")
    logger.info("Downloading and reading artifact")
    artifact = run.use_artifact(
        args.engineer_input_artifact, type='train_data')
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path, low_memory=False)

    curr_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:

        filename = args.engineer_output_artifact  # "engineered_data.csv"
        temp_path = os.path.join(tmp_dir, filename)

        # Save clean df to local machine if desired
        if args.save_engineered_locally is True:
            df2 = df.copy()
            df2.to_csv(os.path.join(
                curr_dir, args.engineer_output_artifact))
        df.to_csv(temp_path)  # might need index=False

        # Create artifact and upload to wandb
        artifact = wandb.Artifact(
            name=args.engineer_output_artifact,
            type=args.engineer_output_type,
            description="csv file of engineered training data",
        )
        artifact.add_file(temp_path)
        logger.info("Logging artifact")
        run.log_artifact(artifact)

        artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Engineering data")
    parser.add_argument(
        "--engineer_input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--engineer_output_artifact",
        type=str,
        help="Name for the output data frame artifact",
        required=True
    )

    parser.add_argument(
        "--save_engineered_locally",
        type=lambda x: bool(strtobool(x)),
        help='Choose whether to save engieered data frame to a local file',
        required=True
    )

    parser.add_argument(
        "--tfidf_max_features",
        type=float,
        help="maximum features to use in TfidfVectorizer",
        required=True
    )
    parser.add_argument(
        "--engineer_artifact_type",
        type=str,
        help="Type of the produced artifact",
        required=True
    )

    parser.add_argument(
        "--use_host",
        type=lambda x: bool(strtobool(x)),
        help='Choose whether to use the host_name feature',
        required=True
    )
    parser.add_argument(
        "--use_neighbourhood",
        type=lambda x: bool(strtobool(x)),
        help='Choose whether to use the neighbourhood feature',
        required=True
    )

    parser.add_argument(
        "--min_date",
        type=float,
        help='lower range for "last_review" feature outliers',
        required=True
    )

    parser.add_argument(
        "--max_date",
        type=float,
        help='upper range for "last_review" feature outliers',
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

    parser.add_argument(
        "--min_listings",
        type=float,
        help="minimum listings lower param",
        required=True
    )

    parser.add_argument(
        "--max_listings",
        type=float,
        help="minimum listings upper param",
        required=True
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
