import argparse
import logging
import os

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="process_data")

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact,
                                type='raw_data')
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path, low_memory=False)

    # Drop the duplicates
    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    logger.info("Dropping outliers")
    print(list(df.columns))
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    filename = args.output_name  # "preprocessed_data.csv"

    idx = df['longitude'].between(-74.25, -73.50) &\
        df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
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
