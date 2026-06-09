"""
finger_scaling_laws.py

The code for running the evaluation pipeline with one, two or three actuators in the
training set and save the results to a dataframe for further analysis.

usage (from project root):
    - uv run src/finger_scaling_laws.py

"""

import pandas as pd
import evaluation

from itertools import combinations


def non_empty_subsets(lst):
    """
    Find all non empty subsets of a list
    """
    subsets = []
    for r in range(1, len(lst) + 1):
        subsets.extend(combinations(lst, r))
    return subsets


if __name__ == "__main__":
    # load the z-score normalized samples
    df = pd.read_parquet("./analysis_data/all_embeddings.parquet")
    df = df[df["task"].isin(["regression"])]

    finger_ids = df["finger_id"].unique()

    subsets = non_empty_subsets(finger_ids)
    results = []
    for train_subset in subsets[:-1]:
        # concat the ids from the training set and override the finger_id with the concatenation in the data
        # this is necessary for the downstream script to work
        train_id = "".join(train_subset)
        df_train = df.copy()
        df_train.loc[df_train["finger_id"].isin(train_subset), "finger_id"] = train_id

        # run the evaluation
        _, df_result = evaluation.run_downstream_evaluation(
            all_results_df=df_train,
            num_evaluations=50,
            is_scaling_law=True,
            train_id=train_id,
        )

        # append the results
        df_result["train_size"] = len(train_subset)
        df_result["train_set"] = train_id
        results.append(df_result)

    all_results_df = pd.concat(results, axis=0)
    all_results_df.to_parquet("./analysis_data/finger_scaling_laws.parquet")
    print(all_results_df)
