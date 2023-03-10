
# setuptools error can be fixed with conda update --force conda
import mlflow
import os
import hydra
import json
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name='config', config_path='config', version_base='2.5')
def go(config: DictConfig):
    """
    Run MLflow project. From main directory, download step can be run using:
    mlflow run . -P hydra_options="main.execute_steps='download'"
    Can be run in github using:
    mlflow run https://github.com/DyRutter/rental_prices.git -v 1.0.1 -P
    hydra_options="data.sample='sample2.csv'"
    """
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()
    if isinstance(config["main"]["execute_steps"], str):
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = config["main"]["execute_steps"]

    # Make sure github file names match the names of the steps, otherwise
    # throws private error
    if "download" in steps_to_execute:
        _ = mlflow.run(
            f"{config['main']['download_repository']}",
            version="main",
            entry_point="main",
            parameters={
                    # URL to raw data sample
                    "file_url": config["data"]["file_url"],
                    # "raw_data.csv",
                    "artifact_name": config["data"]["raw_data_name"],
                    # "raw_data"
                    "artifact_type": config["data"]["raw_data_type"],
                    # "Raw file as downloaded"
                    "artifact_description": config["data"]
                    ["raw_data_description"]})

    if "basic_cleaning" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "basic_cleaning"),
            entry_point="main",
            parameters={
                # "raw_data.csv:latest",
                "input_artifact": config["data"]["raw_data_artifact"],
                "output_name": config["data"]["preprocessed_data"],
                "output_type": "preprocessed_data",
                "output_description": "Data with preprocessing applied",
                "min_price": config["data"]["min_price"],
                "max_price": config["data"]["max_price"]})

    if "check_data" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            entry_point="main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                # "preprocessed_data.csv:latest",
                "sample_artifact": config["data"]["preprocessed_data_latest"],
                "kl_threshold": config["data"]["kl_threshold"],
                "min_price": config["data"]["min_price"],
                "max_price": config["data"]["max_price"]})

    # if "EDA" in steps_to_execute:
    #   _ = mlflow.run(
    #      os.path.join(root_path, "EDA"),
    #      entry_point="main")

    if "segregate" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            entry_point="main",
            parameters={
                # preprocessed_data.csv:latest",
                "input_artifact": config["data"]["preprocessed_data_latest"],
                "artifact_root": "data",
                "artifact_type": "segregated_data",
                "test_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"]})

    if "random_forest" in steps_to_execute:
        # NOTE: we need to serialize the random forest configuration into JSON
        model_config = os.path.abspath("random_forest_config.yml")
        rf_config = os.path.abspath("rf_config.json")
        with open(rf_config, "w+") as fp:
            json.dump(
                dict(
                    config["random_forest_pipeline"]["random_forest"].items()),
                fp)
        with open(model_config, "w+", encoding="utf-8") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))
        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            entry_point="main",
            parameters={
                # "data_train.csv:latest",
                "trainval_artifact": config["data"]["train_data"],
                "rf_config": rf_config,
                "output_artifact":\
                config["random_forest_pipeline"]["output_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["test_size"],
                "stratify_by": config["data"]["stratify_by"],
                "max_tfidf_features":\
                config["random_forest_pipeline"]["max_tfidf_features"],
                "save_locally":\
                config["random_forest_pipeline"]["save_locally"]})

    if "test_regression_model" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(
                root_path,
                "test_regression_model"),
            entry_point="main",
            parameters={
             "mlflow_model":
             f"{config['random_forest_pipeline']['output_artifact']}:latest",
             "test_dataset": config["data"]["test_data"]})


if __name__ == "__main__":
    go()
