"""
create_plots.py

Contains the code for creating the different plots for:
    - the transferability baseline
    - the finger scaling law
    - the effect of the normalization
    - the effect of the affine transformation

All plots are created as vector graphics (.pdf).

usage:
    # from repository root directory
    uv run src/create_plots.py

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_bar_plot(
    df_plot,
    xticklabels,
    file_path,
    score,
    title=None,
    x_label=None,
):
    """
    Generates a bar plot from the data in `df_plot`.
    Two bars per group in xticklabels are created, one for the in-distribution and one for the out-of-distribution actuators.

    - `df_plot`: a pandas DataFrame that should have the columns {this|other}_{r2|mean_absolute_error}.{mean|std}
    - `file_path`: location to save the resulting plot too, the file ending specifies the resulting format.
    - `score`: the score to use for this plot, either r2 or mae
    - `title`: default None, if no title is provided it is generated from the `score` and `x_label`
    - `x_label`: default None, labels the x-axis, and is omitted if no labels are provided
    """
    # set the score so it is compatible with the dataframe format
    if score == "mae":
        score = "mean_absolute_error"
        score_name = "MAE"
    elif score == "r2":
        score_name = "R²"
    else:
        raise NotImplementedError

    # get the number of groups on the x axis
    x = np.arange(len(xticklabels))
    width = 0.35

    colors = ["#4C72B0", "#DD8452"]

    fig_width_in = 5.6
    fig_height_in = 7.5
    dpi = 600
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)

    # This {score}
    bar_1 = ax.bar(
        x - width / 2,
        df_plot[f"this_{score}.mean"].values,
        width,
        yerr=df_plot[f"this_{score}.std"].values,
        capsize=5,
        label="same actuator",
        color=colors[0],
    )

    # Other {score}
    bar_2 = ax.bar(
        x + width / 2,
        df_plot[f"other_{score}.mean"].values,
        width,
        yerr=df_plot[f"other_{score}.std"].values,
        capsize=5,
        label="other actuators",
        color=colors[1],
    )

    ax.bar_label(bar_1, fmt="%.2f", padding=2)
    ax.bar_label(bar_2, fmt="%.2f", padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    if x_label is None:
        x_label = " "
    ax.set_xlabel(x_label)

    ax.set_ylabel(f"{score_name} (mean ± std)")

    if title is None:
        title = f"{score_name} by {x_label}"

    fig.suptitle(title, fontweight="bold")

    ax.legend(loc="center left")
    plt.tight_layout()
    plt.savefig(
        file_path,
        dpi=dpi,
    )


def create_baseline_and_norm_plots():
    """
    Creates the plots:
        - baseline transferability
        - effect of normalisation

    Requires the corresponding datasets to be available at <project_root>/analysis_data/normalisation/...
    """

    df_nonorm = pd.read_json(
        "./analysis_data/test_results_regression_no_norm.json",
        orient="split",
    )
    df_zscore = pd.read_json(
        "./analysis_data/test_results_regression_z-score.json",
        orient="split",
    )
    df_minmax = pd.read_json(
        "./analysis_data/test_results_regression_min-max.json",
        orient="split",
    )

    df_nonorm.columns = [
        f"{col}.{value}" if value != "" else col for col, value in df_nonorm.columns
    ]
    df_zscore.columns = [
        f"{col}.{value}" if value != "" else col for col, value in df_zscore.columns
    ]

    df_minmax.columns = [
        f"{col}.{value}" if value != "" else col for col, value in df_minmax.columns
    ]
    df_nonorm = df_nonorm.loc[df_nonorm["other_mean_absolute_error.mean"].idxmin()]
    df_zscore = df_zscore.loc[df_zscore["other_mean_absolute_error.mean"].idxmin()]
    df_minmax = df_minmax.loc[df_minmax["other_mean_absolute_error.mean"].idxmin()]

    df_plot = (
        pd.concat(
            [
                df_nonorm.rename("Not Normalized"),
            ],
            axis=1,
        )
        .T.reset_index(names="method")
        .set_index("method")
    )

    create_bar_plot(
        df_plot=df_plot,
        xticklabels=["baseline"],
        file_path="./plots/baseline/baseline_regression_mae.pdf",
        score="mae",
        title="MAE Same vs. Other Actuator(s)",
    )

    create_bar_plot(
        df_plot=df_plot,
        xticklabels=["baseline"],
        file_path="./plots/baseline/baseline_regression_r2.pdf",
        score="r2",
        title="R² Same vs. Other Actuator(s)",
    )

    df_plot = (
        pd.concat(
            [
                df_nonorm.rename("Not Normalized"),
                df_minmax.rename("Min-Max"),
                df_zscore.rename("Z-Score"),
            ],
            axis=1,
        )
        .T.reset_index(names="method")
        .set_index("method")
    )

    create_bar_plot(
        df_plot=df_plot,
        x_label="Normalization Method",
        xticklabels=["None", "Min-Max", "Z-Score"],
        file_path="./plots/norm/mae_norm_no_norm.pdf",
        score="mae",
    )

    create_bar_plot(
        df_plot=df_plot,
        x_label="Normalization Method",
        xticklabels=["None", "Min-Max", "Z-Score"],
        file_path="./plots/norm/r2_norm_no_norm.pdf",
        score="r2",
    )


def create_finger_scaling_law_plots():
    """
    Creates the plot:
        - finger scaling law

    Requires the corresponding dataset to be available at <project_root>/analysis_data/...
    """
    df = pd.read_parquet("./analysis_data/finger_scaling_laws.parquet")
    df = df.reset_index(drop=True)

    df.columns = [f"{col}.{value}" if value != "" else col for col, value in df.columns]
    df = df.groupby(["model", "train_size"]).mean(numeric_only=True)
    df = df.loc[df.groupby(["train_size"])["other_mean_absolute_error.mean"].idxmin()]

    df_mean = df.groupby(["train_size"]).mean(numeric_only=True)
    df_plot = df_mean.reset_index()

    create_bar_plot(
        df_plot=df_plot,
        x_label="#Actuators in Train Set",
        xticklabels=["1", "2", "3"],
        file_path="./plots/finger_scaling_law/r2_by_train_size.pdf",
        score="r2",
    )

    create_bar_plot(
        df_plot=df_plot,
        x_label="#Actuators in Train Set",
        xticklabels=["1", "2", "3"],
        file_path="./plots/finger_scaling_law/mae_by_train_size.pdf",
        score="mae",
    )


def create_affine_plots():
    """
    Creates the plot:
        - effect of affine transform

    Requires the corresponding datasets to be available at <project_root>/analysis_data/...
    """
    df_nonorm = pd.read_json(
        "./analysis_data/test_results_regression_no_norm.json", orient="split"
    )
    df_zscore = pd.read_json(
        "./analysis_data/test_results_regression_z-score.json", orient="split"
    )
    df_nonorm.columns = [
        f"{col}.{value}" if value != "" else col for col, value in df_nonorm.columns
    ]
    df_zscore.columns = [
        f"{col}.{value}" if value != "" else col for col, value in df_zscore.columns
    ]
    df_nonorm = df_nonorm.loc[df_nonorm["other_mean_absolute_error.mean"].idxmin()]
    df_zscore = df_zscore.loc[df_zscore["other_mean_absolute_error.mean"].idxmin()]

    df = pd.read_parquet("./analysis_data/affine_transformations_results.parquet")
    df = df.reset_index(drop=True)
    df.columns = [f"{col}.{value}" if value != "" else col for col, value in df.columns]
    df = df.loc[df.groupby(["from", "to"])["other_mean_absolute_error.mean"].idxmin()]

    df_plot = df.mean(numeric_only=True)
    df_plot = (
        pd.concat(
            [
                df_nonorm.rename("nonnorm"),
                df_zscore.rename("z-score"),
                df_plot.rename("affine"),
            ],
            axis=1,
        )
        .T.reset_index(names="method")
        .set_index("method")
    )

    create_bar_plot(
        df_plot=df_plot,
        x_label="Adaption Method",
        xticklabels=["None", "Z-Score", "Affine"],
        file_path="./plots/affine_transform/r2_affine_compared_to_reference.pdf",
        score="r2",
    )
    create_bar_plot(
        df_plot=df_plot,
        x_label="Adaption Method",
        xticklabels=["None", "Z-Score", "Affine"],
        file_path="./plots/affine_transform/mae_affine_compared_to_reference.pdf",
        score="mae",
    )


def set_publication_style():
    """
    Sets global styling defaults for matplotlib
    """
    plt.rcParams.update({
        "font.size": 17,  # base font
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


if __name__ == "__main__":
    print("Creating output folders if they don't exist...")
    os.makedirs("./plots/affine_transform", exist_ok=True)
    os.makedirs("./plots/finger_scaling_law", exist_ok=True)
    os.makedirs("./plots/baseline", exist_ok=True)
    os.makedirs("./plots/norm", exist_ok=True)

    print("Creating and saving plots...")

    set_publication_style()
    create_baseline_and_norm_plots()
    create_finger_scaling_law_plots()
    create_affine_plots()

    print("Done!")
