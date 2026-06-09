"""
fit_affine_transforms.py

Contains the code for fitting an affine transform, the AffineEmbedder class for encapsulating this transform
and the code to evaluate the effect of the affine transformation.

usage (from project root):
    - uv run src/fit_affine_transforms.py

"""

import os

import pandas as pd
import numpy as np

import evaluation


def fit_affine_xy(X, Y):
    """
    Fits y ≈ A x + b using least squares.

    X: (n_samples, d)
    Y: (n_samples, d)
    Returns:
        A: (d, d)
        b: (d,)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    n, d = X.shape
    assert Y.shape == (n, d)

    # Augment X with ones for bias
    X_aug = np.hstack([X, np.ones((n, 1))])  # (n, d+1)

    # Solve X_aug @ W ≈ Y
    W, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)

    A = W[:-1].T  # (d, d)
    b = W[-1]  # (d,)

    return A, b


class AffineEmbedder:
    """
    A wrapper class that encapsulates an affine transform and can be called to transform data.

    """

    def __init__(self, X_from, X_to):
        """
        Creates a new AffineEmbedder object.
        Fits an affine transform from `X_from` to `X_to`, both numpy arrays.

        fixes the random seed for numpy
        """
        np.random.seed(0)
        self.A, self.b = fit_affine_xy(X_from, X_to)

    def __call__(self, X):
        """
        Transforms and returns the input data `X` (numpy array) using the learned affine transformation:
            X' = AX + b
        """
        return X @ self.A.T + self.b


def evaluate_from_to(df_mean, df_eval, finger_from, finger_to):
    """
    Fits an affine transform from `finger_from` to `finger_to` using the per label means
    in `df_mean` and then evaluate it using `df_eval` which contains all samples
    - `df_mean`: per label means over all samples for a finger
    - `df_eval`: all samples for all fingers
    - `finger_from`: string id of the "from" finger
    - `finger_to`: string id of the "to" finger
    """
    # ensure consistent label sorting
    df_mean = df_mean.sort_values(by="labels")

    # create masks that return only the average per label for the regression tasks for both fingers
    mask_from = df_mean["finger_id"].isin([finger_from]) & df_mean["task"].isin([
        "regression"
    ])
    mask_to = df_mean["finger_id"].isin([finger_to]) & df_mean["task"].isin([
        "regression"
    ])

    # create masks that return all the datapoints for the two fingers and the regression task
    mask_from_eval = df_eval["finger_id"].isin([finger_from]) & df_eval["task"].isin([
        "regression"
    ])
    mask_to_eval = df_eval["finger_id"].isin([finger_to]) & df_eval["task"].isin([
        "regression"
    ])

    # Turn the per-label mean of the mean frequencies into numpy arrays
    X = np.vstack(df_mean.loc[mask_from, "embeddings"].values)
    Y = np.vstack(df_mean.loc[mask_to, "embeddings"].values)

    # fit the affine embedding
    embedder = AffineEmbedder(X, Y)

    # embed all samples from the "from" finger to the space of the "to" finger
    X_eval = np.vstack(df_eval.loc[mask_from_eval, "embeddings"].values)
    embeddings = embedder(X_eval)

    # overwrite the samples with their transformed samples in the dataframe that contains all samples
    df_eval.loc[mask_from_eval, "embeddings"] = pd.Series(
        list(embeddings), index=df_eval.index[mask_from_eval]
    )

    # create a dataframe that only consists the samples from the two relevant fingers
    # in their transformed form
    all_results_df = pd.concat(
        [df_eval.loc[mask_from_eval], df_eval.loc[mask_to_eval]], axis=0
    )

    # save a pca plot of all samples after the projection
    fig = evaluation.embedding_pca(
        all_results_df, "regression", f"Affine from {finger_from} to {finger_to}"
    )
    fig.savefig(f"./plots/affine_per_transform/from_{finger_from}_to{finger_to}.png")

    # run the evaluation (same as the ml pipeline) on the transformed data
    _, out = evaluation.run_downstream_evaluation(
        all_results_df, num_evaluations=100, is_affine=True, finger_to=finger_to
    )
    return out


def mean_lists(series_of_lists):
    """
    helper function to stack lists from a dataframe
    """
    # Stack the lists into a 2D array and take mean along axis=0
    return np.mean(np.stack(series_of_lists), axis=0)


if __name__ == "__main__":
    os.makedirs("./plots/affine_per_transform", exist_ok=True)
    df_eval = pd.read_parquet("./analysis_data/all_embeddings.parquet")

    # Take the mean per label, finger_id and task
    df = df_eval.groupby(["labels", "finger_id", "task"], as_index=False).agg({
        "embeddings": mean_lists
    })

    # iterate through all pairs of fingers
    fingers = [
        "030",
        "032",
        # "037",
        # "035",
    ]  # only use the fingers that have a regression dataset
    dfs = []
    for finger_from in fingers:
        for finger_to in fingers:
            if finger_from == finger_to:
                # skip identity mappings
                continue

            # fit a transform and run the evaluation on it for this finger pair
            df_out = evaluate_from_to(df.copy(), df_eval, finger_from, finger_to)

            # the finger pair ids to the output
            df_out["from"] = finger_from
            df_out["to"] = finger_to

            dfs.append(df_out)

    # concat and save results
    df_results = pd.concat(dfs, axis=0)
    df_results.to_parquet("./analysis_data/affine_transformations_results.parquet")
