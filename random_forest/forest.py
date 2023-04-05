import argparse
import tempfile
import logging
import os
import shutil
import joblib
import wandb
import mlflow
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # Instantiate run
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config, 'r') as fp:
        rf_config = json.loads(fp.read())
    run.config.update(rf_config)

    # Fix the random seed for the Random Forest
    rf_config['random_state'] = args.random_seed

    # Get local path to train/validation data
    trainval_local_path = run.use_artifact(
        args.trainval_artifact).file()

    # Get features and labels
    X = pd.read_csv(trainval_local_path)
    for_img = X.copy()
    y = X.pop("price")

    # Log minimum and maximum prices
    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=X[args.stratify_by] if args.stratify_by != "null" else None,
        random_state=args.random_seed,
        shuffle=True)

    # Instantiate regresssor and fit to data
    logger.info("Fitting")
    forest = RandomForestRegressor(**rf_config)
    forest.fit(X_train, y_train)

    # Compute r2 and MAE
    logger.info("Scoring")
    r_squared = forest.score(X_val, y_val)
    y_pred = forest.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    logger.info(f"R2 Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    # Log as metadata in artifact
    rf_config["R2"] = r_squared
    rf_config["MAE"] = mae

    # Export model to wandb
    logger.info("Exporting model")
    # Save model package in the MLFlow sklearn format to "random_forest_dir"
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    # Infer a signature for the model
    signature = infer_signature(X_val, y_pred)
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "random_forest_dir")
        mlflow.sklearn.save_model(
          forest,
          path=export_path,
          signature=signature,
          input_example=X_val.iloc[:2],
          serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

        # Upload model as an artifact
        artifact = wandb.Artifact(args.output_artifact,
                                  type="model_export",
                                  description="RF model",
                                  metadata=rf_config)
        artifact.add_dir(export_path)
        run.log_artifact(artifact)
        artifact.wait()

    # Save r_squared and mae under the "r2" and "mae" respectively
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(forest, for_img)

    # Upload feture importance visualization to wandb
    run.log({"feature_importance": wandb.Image(fig_feat_imp)})

    # Save figure and compressed model to local machine if desired
    if args.save_model_locally is True:
        filepath = os.path.join(os.getcwd(), 'model_file')
        if not os.path.isdir(filepath):
            os.umask(0)
            os.makedirs(filepath)
        fig_feat_imp.savefig('./model_file/rf_feature_importances.png')
        joblib.dump(forest, './model_file/rf_model.pkl', compress=6)
        logger.info("files saved to local machine")


def plot_feature_importance(model, X):
    """
    Find and graph our model's feature importances.
    Recreate our initial features in three steps:
        1) Collect the feature importance for all non-nlp features
        2) Merge all nlp importances into single 'name' feature
        3) Merge all 'neighbourhood_group' cat variables into a single feature
    Note all importances sum to one, confirming we've used all features
    Input is a model and data_frame X
    """
    # Separating features into lists to get list of non-nlp and non-cats
    nameless = [col for col in X.columns if "Name_" not in col]
    groupless = [feat for feat in nameless if "neighbourhood_" not in feat]
    neighless = [feat for feat in groupless if "neigh_" not in feat]
    feat_names = [feat for feat in neighless if "Host_" not in feat]

    # Indexing normal features and summing their importance for reference
    feat_idx = len(feat_names)
    feat_imp = model.feature_importances_[:len(feat_names)]
    # feat_imp_sum = sum(feat_imp)
    # print (feat_imp_sum)

    # Index and sum NLP feature sum importances across all TF-IDF dimensions
    name_idx = len([col for col in X.columns if "Name_" in col])
    name_imp = sum(model.feature_importances_[feat_idx: feat_idx + name_idx])
    # print (name_imp)

    # Sum importance of categorical features into global group cluster
    group_imp = sum(model.feature_importances_[feat_idx + name_idx:])
    # print (group_imp)

    # Create new list of all features
    all_feats = feat_names + ["name", "neighbourhood_group"]
    # Create new list of all feature importances
    feat_imp = np.append(feat_imp, name_imp)
    feat_imp = np.append(feat_imp, group_imp)

    # Plot importances
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(
        range(
            feat_imp.shape[0]),
        feat_imp,
        color="r",
        align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(all_feats), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset."
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed "
        "to the scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--max_tfidf_features",
        help="Maximum number of words to consider for the TFIDF",
        default=10,
        type=int
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    parser.add_argument(
        "--save_model_locally",
        type=str,
        help="Set to true if you want to save figures in local folder",
        required=False,
        default=True,
    )

    args = parser.parse_args()

    go(args)
