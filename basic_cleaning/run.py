import argparse
import logging
import os
from distutils.util import strtobool
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def drop_features(df):
    """
    Features chosen for dropping from Pandas profile
    Drop 'id' and 'host_id" for cardinality
    Drop 'number_of_reviews' and 'reviews_per_month' b/c of high correlation
    """
    df = df.drop(['id', 'host_id'], axis=1)
    return df


def drop_useless(df):
    """
    Drop duplicates
    Drop any samples missing the 'price' input
    """
    df = df.drop_duplicates().reset_index(drop=True)
    df.dropna(subset=['price'], inplace=True)
    return df


def go(args):

    # Instantiate wandb, run, and get raw data artifact
    run = wandb.init(job_type="process_data")
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact, type='raw_data')
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path, low_memory=False)

    # Drop unused features, duplicates, and outliers
    logger.info("Dropping unused features")
    df = drop_features(df)
    logger.info("Dropping duplicates and rows missing 'price'")
    df = drop_useless(df)

    filename = args.output_name

    # Save clean df to local machine if desired
    if args.save_clean_locally is True:
        df2 = df.copy()
        local = 'basic_data.csv'
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

    # True values are y, yes, t, true, on and 1;
    # False values are n, no, f, false, off and 0
    # Will raise ValueError if input argument is not of proper type
    parser.add_argument(
        "--save_clean_locally",
        type=lambda x: bool(strtobool(x)),
        help='Choose whether or not to save clean data frame to local file',
        required=True
    )

    args = parser.parse_args()

    go(args)
