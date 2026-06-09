"""
evaluation.py

Contains all code for the evaluation step of the pipeline.
For a representation, we fit different models on one finger, and evaluate the transferability to new fingers.

"""

import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor

# config obtained from AutoML run in src/explore_automl.py
regressor = LGBMRegressor(
    colsample_bytree=np.float64(0.763007791741338),
    learning_rate=np.float64(0.16645809713264254),
    max_bin=1023,
    min_child_samples=6,
    n_estimators=47,
    n_jobs=-1,
    num_leaves=12,
    reg_alpha=0.0009765625,
    reg_lambda=np.float64(0.10626868868028042),
    verbose=-1,
)

## All classification models that we evaluate on
classification_models = list(
    zip(
        ["knnclassifier", "logisticregression"],  # names
        [KNeighborsClassifier, LogisticRegression],  # models
    )
)
## The evaluation metrics for the classification tasks
classification_metrics = list(
    zip(
        ["accuracy"],
        [accuracy_score],
    )
)

## All regression models that we evaluate on
regression_models = list(
    zip(
        ["knnregressor", "ridgeregressor", "lgbm"],  # names
        [
            KNeighborsRegressor,
            Ridge,
            lambda: regressor,
        ],  # models
    )
)
## The evaluation metrics for the regression task
regression_metrics = list(
    zip(
        ["r2", "mean_absolute_error"],
        [r2_score, mean_absolute_error],
    )
)


def score_results(task, prefix, y_true, y_pred):
    """
    Calculates all relevant metrics for `y_true` and `y_pred`
    - `task`: "regression" or "classification"
    - `prefix`: string prefix to add to the returned metrics
    - `y_true`: labels
    - `y_pred`: predictions

    Returns a dict of the metrics
    """
    results = {}
    if task == "regression":
        metrics = regression_metrics
    else:
        metrics = classification_metrics

    for name, metric in metrics:
        results[f"{prefix}_{name}"] = metric(y_true, y_pred)

    return results


def fit_and_evaluate(
    model,
    model_name,
    task,
    finger_id,
    x_train,
    y_train,
    x_test,
    y_test,
    x_other,
    y_other,
    use_pca=False,
):
    """
    Fit a model on a specific train and test set, evaluate it and return the results as a dictionary.
    - `model`: sklearn compatible model
    - `model_name`: identifier that is added to the return dictionary
    - `task`: "regression" or "classification"
    - `finger_id`: the id of the finger that the train and test samples were recorded on
    - `x_train`: train samples
    - `y_train`: train labels
    - `x_test`: test samples
    - `y_test`: test labels
    - `x_other`: test samples from other actuators
    - `y_other`: test labels from other actuators
    - `use_pca`: if True, add a PCA step before the model to make it work in a lower dimensional space
    Returns a dictionary with `model_name`, `task`, `finger_id` and the resulting scores
    """
    # construct a pipeline with PCA preprocessing if applicable
    if use_pca:
        model = Pipeline(steps=[("pca", PCA()), ("model", model())])
    else:
        model = model()

    # fit the model/pipeline
    model.fit(x_train, y_train)

    # make the predictions for this finger and the other fingers
    y_this_pred = model.predict(x_test)
    y_other_pred = model.predict(x_other)

    # score the predictions
    score_this = score_results(
        task,
        prefix="this",
        y_true=y_test,
        y_pred=y_this_pred,
    )
    score_other = score_results(
        task,
        prefix="other",
        y_true=y_other,
        y_pred=y_other_pred,
    )

    # return the results
    return {
        "finger_id": finger_id,
        "model": model_name,
        "task": task,
        **score_this,
        **score_other,
    }


def run_downstream_evaluation(
    all_results_df,
    num_evaluations=50,
    is_affine=False,
    finger_to=None,
    is_scaling_law=False,
    train_id=None,
    use_pca=False,
):
    """
    Run the transferability evaluation for a dataframe of representations.
    - `all_results_df`: the dataframe that contains the embeddings
    - `num_evaluations`: how many random seeds to rerun this evaluation with
    - `is_affine`: if we are evaluating an affine transformation
    - `finger_to`: only for the affine transformation, the finger we are mapping to using the affine transform
    - `is_scaling_law`: if we are evaluating the finger scaling law
    - `train_id`: only scaling law: the finger_ids of the fingers in the train set
    - `use_pca`: whether to use PCA to reduce dimensionality before fitting models on the embeddings
    returns:
        - df_classification, df_regression
        the results for the classification and regression tasks

    """
    # create pairs of fingers and tasks to run the evaluation on
    if is_affine:
        unique_pairs = all_results_df[all_results_df["finger_id"].isin([finger_to])][
            ["finger_id", "task"]
        ].drop_duplicates()
    if is_scaling_law:
        unique_pairs = pd.DataFrame({"id": [train_id], "task": "regression"})
    else:
        # use all possible pairs (default)
        unique_pairs = all_results_df[["finger_id", "task"]].drop_duplicates()

    results_regression = []
    results_classification = []
    for finger_id, task in unique_pairs.itertuples(index=False, name=None):
        # dataframe of samples of the finger(s) that we train on
        df_this_finger = all_results_df[
            (all_results_df["finger_id"] == finger_id)
            & (all_results_df["task"] == task)
        ]

        # dataframe of samples for the other fingers
        df_other_fingers = all_results_df[
            (all_results_df["finger_id"] != finger_id)
            & (all_results_df["task"] == task)
        ]

        # create numpy arrays from dataframes
        x_this = np.vstack(df_this_finger["embeddings"].values)
        y_this = df_this_finger["labels"].values

        x_other = np.vstack(df_other_fingers["embeddings"].values)
        y_other = df_other_fingers["labels"].values

        # pick the correct models to train on
        if task == "classification":
            models = classification_models
            results = results_classification
        else:
            models = regression_models
            results = results_regression

        # loop through the random seeds and fit the models
        for i in tqdm.tqdm(
            range(0, num_evaluations), desc="looping through random seeds"
        ):
            x_train, x_test, y_train, y_test = train_test_split(
                x_this,
                y_this,
                test_size=0.2,
                stratify=y_this,
                random_state=1337 + i,
            )
            for name, model in models:
                result = fit_and_evaluate(
                    model,
                    name,
                    task,
                    finger_id,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    x_other,
                    y_other,
                    use_pca=use_pca,
                )
                results.append(result)

    # aggregate the results
    df_regression = (
        pd.DataFrame(results_regression)
        .drop("finger_id", axis=1)
        .groupby(["model", "task"])
        .agg(["mean", "std"])
    ).reset_index()

    if results_classification != []:
        df_classification = (
            pd.DataFrame(results_classification)
            .drop("finger_id", axis=1)
            .groupby(["model", "task"])
            .agg(["mean", "std"])
        ).reset_index()
    else:
        df_classification = pd.DataFrame([])

    return df_classification, df_regression


def embedding_pca(all_results_df, task, title):
    """
    Create a pca visualisation of the embeddings for a specific task:
        - `all_results_df`: dataframe that contains all the embeddings
        - `task`: "regression" or "classification"
        - `title`: title for the plot

    returns a PCA plot
    """
    # concat dfs with finger id
    df = all_results_df[all_results_df["task"] == task].copy()
    pca = PCA(n_components=2, random_state=42)
    embeddings = np.array(df["embeddings"].tolist())
    principal_components = pca.fit_transform(embeddings)

    df["pc1"] = principal_components[:, 0]
    df["pc2"] = principal_components[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="pc1",
        y="pc2",
        hue="labels",
        style="finger_id",
        s=60,
        alpha=0.8,
        palette="Set1",
    )
    plt.title(title)
    return plt.gcf()


def embedding_tsne(all_results_df, task, title):
    """
    Create a T-SNE visualisation of the embeddings for a specific task:
        - `all_results_df`: dataframe that contains all the embeddings
        - `task`: "regression" or "classification"
        - `title`: title for the plot

    returns a T-SNE plot
    """
    # concat dfs with finger id
    df = all_results_df[all_results_df["task"] == task].copy()
    pca = TSNE(n_components=2, random_state=42)
    embeddings = np.array(df["embeddings"].tolist())
    principal_components = pca.fit_transform(embeddings)

    df["c1"] = principal_components[:, 0]
    df["c2"] = principal_components[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="c1",
        y="c2",
        hue="labels",
        style="finger_id",
        s=60,
        alpha=0.8,
        palette="Set1",
    )
    plt.title(title)
    return plt.gcf()
