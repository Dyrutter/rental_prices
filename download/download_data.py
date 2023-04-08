import argparse
import logging
import pathlib
import wandb
import requests
import tempfile
import os
import pandas as pd
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # Derive the base name of the data file from the URL
    basename = pathlib.Path(args.file_url).name.split("?")[0].split("#")[0]
    # Download file, streaming so we can download files larger than
    # the available memory. Named temporary file gets destroyed at end so
    # nothing is left behind and the file gets removed even in case of errors
    logger.info(f"Downloading {args.file_url} ...")
    data_filename = os.path.join(os.getcwd(), f"{args.artifact_name}.csv")
    with tempfile.NamedTemporaryFile(mode='wb+') as fp:
        logger.info("Creating run")
        with wandb.init(job_type="download_data") as run:
            # Download the file streaming and write to open temp file
            with requests.get(args.file_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)
            fp.flush()  # ensure file written to disk before uploading to w&B
            logger.info("Creating artifact")
            artifact = wandb.Artifact(
                name=args.artifact_name,
                type=args.artifact_type,
                description=args.artifact_description,
                metadata={'original_url': args.file_url})
            artifact.add_file(fp.name, name=basename)
            logger.info("Logging artifact")
            run.log_artifact(artifact)
            artifact.wait()

    with wandb.init(job_type='download_data') as run:
        artifact = run.use_artifact(args.artifact_name)
        artifact_path = artifact.file()
        df = pd.read_csv(artifact_path, low_memory=False)
        df.to_csv(data_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to W&B",
        fromfile_prefix_chars="@")
    # parser.add_argument
    # ("--sample", type=str, help="sample csv file", required=True)
    parser.add_argument(
        "--file_url", type=str, help="File URL", required=True)
    parser.add_argument(
        "--artifact_name", type=str, help="Name for artifact", required=True)
    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type",
        required=True)
    parser.add_argument("--artifact_description", type=str,
                        help="Description for the new artifact", required=True)
    args = parser.parse_args()
    go(args)
