#!/usr/bin/env python
import argparse
import logging
import os
from distutils.util import strtobool
import tempfile
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # Initiate run and get processed_data artifact
    run = wandb.init(job_type="split_data")
    logger.info("Downloading and reading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    # Remember to pass index_col argument to prevent creating a new column
    df = pd.read_csv(artifact_path, low_memory=False, index_col=[0])

    # Split in model_dev/test, then divide model_dev in train and validation
    logger.info("Splitting data into train, val and test")
    splits = {}
    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Save the artifacts. Use a temporary directory to remove traces
    curr_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:

        for split, df in splits.items():

            # Make the artifact name from the root plus "train" and "test"
            artifact_name = f"{args.split_artifact_root}_{split}.csv"

            # Get the path on disk within the temp directory
            temp_path = os.path.join(tmp_dir, artifact_name)

            logger.info(f"Uploading the {split} dataset to {artifact_name}")

            # Save locally if desired
            if args.save_split_locally is True:
                df2 = df.copy()
                df2.to_csv(os.path.join(
                    curr_dir, f"{args.split_artifact_root}_{split}.csv"))

            # Save then upload to W&B
            df.to_csv(temp_path)
            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                description=f"{split} split of dataset {args.input_artifact}",
            )
            artifact.add_file(temp_path)

            logger.info("Logging artifact")
            run.log_artifact(artifact)

            # This waits for the artifact to be uploaded to W&B. If you
            # do not add this, the temp directory might be removed before
            # W&B had a chance to upload the datasets, and the upload
            # might fail
            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--split_artifact_root",
        type=str,
        help="Root names of the two produced artifacts."
             "{root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the produced artifacts",
        required=True
    )

    parser.add_argument(
        "--test_size",
        help="Fraction of dataset or number of items to use in test split",
        type=float,
        required=True
    )

    parser.add_argument(
        "--random_state",
        help="An integer to use to init the random number generator.",
        type=int,
        required=False,
        default=42
    )

    parser.add_argument(
        "--stratify",
        help="The name of a column to use for stratified splitting",
        type=str,
        required=False,
        default='null'
    )

    # True values are y, yes, t, true, on and 1;
    # False values are n, no, f, false, off and 0
    # Will raise ValueError if input argument is not of proper type
    parser.add_argument(
        "--save_split_locally",
        type=lambda x: bool(strtobool(x)),
        help='Choose whether or not to save split data frames to local files',
        required=True
    )
    args = parser.parse_args()

    go(args)
