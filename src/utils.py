"""
utils.py

Contains helper functions for the normalizations, label extraction from the file name, and others.

"""

import os
import torch


LABEL_DICT_CLASSIFICATION = {
    "none": 0,
    "base": 1,
    "middle": 2,
    "tip": 3,
}
"""
maps the string labels for the classification task to numeric values
"""


def get_finger_id_domain_mapping(configs, finger_to_domain={}):
    """
    Update the mapping from finger ids to integers representing their domain.
    - `config`: the config to add to the domains
    - `finger_to_domain`: the dict that maps the finger id to the domain

    returns the updated dictionary
    """
    if not finger_to_domain:
        domain = 0
    else:
        domain = max(finger_to_domain.values()) + 1

    for config in configs:
        finger_id = config["finger_id"]
        if finger_id not in finger_to_domain.keys():
            finger_to_domain[finger_id] = domain
            domain += 1
    return finger_to_domain


def load_paths(dataset_config, data_basedir):
    """
    helper function to get all audio file paths for a dataset from its base folder
    - `dataset`: a dataset configuration object
    - `data_basedir`: the root dir under which the data dirs are stored
    """
    paths = []
    for path in dataset_config["paths"]:
        full_path = os.path.join(data_basedir, path)
        exclude_list = dataset_config["exclude_files"]
        files = [
            os.path.join(full_path, file)
            for file in os.listdir(full_path)
            if os.path.isfile(os.path.join(full_path, file))
            and file not in exclude_list
        ]
        paths += files
    return paths


def get_num_and_label(filename):
    """
    extract the sample number and the label from a file name
    - `filename`: the file name to extract the information from

    source: acoustic sensing starter kit
    https://git.tu-berlin.de/rbo/robotics/acoustic_sensing_starter_kit

    exact file:
        https://git.tu-berlin.de/rbo/robotics/acoustic_sensing_starter_kit/-/blob/master/B_train.py
    """
    try:
        # remove file extension
        name = os.path.splitext(filename)[0]
        # remove initial number
        name = name.split("_")
        num = int(name[0])
        label = "_".join(name[1:])
        return num, label
    except ValueError:
        # filename with different formatting. ignore.
        print("Value error")
        print(filename)
        return -1, None


def load_paths_and_labels(dataset, data_basedir):
    """
    helper function to get all audio file paths for a dataset from its base folder
    and extract the labels from the file_name
    - `dataset`: a dataset configuration object
    - `data_basedir`: the root dir under which the data dirs are stored
    """
    paths = []
    labels = []
    for path in dataset["paths"]:
        full_path = os.path.join(data_basedir, path)
        exclude_list = dataset["exclude_files"]
        files = [
            os.path.join(full_path, file)
            for file in os.listdir(full_path)
            if os.path.isfile(os.path.join(full_path, file))
            and file not in exclude_list
        ]
        paths += files
        labels += [
            get_num_and_label(file)[1]
            for file in os.listdir(full_path)
            if os.path.isfile(os.path.join(full_path, file))
            and file not in exclude_list
        ]
    return paths, labels


def compute_per_freq_mean_std(dataset):
    """
    Computes the mean and std for each frequency in `dataset`.
    - `dataset`: the dataset to find the values for mean and std in
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4
    )

    sum_ = None
    sum_sq = None
    count = 0

    for batch in loader:
        # batch: (B, freq)
        xs = batch["x"]
        if sum_ is None:
            freq_bins = xs.shape[1]
            sum_ = torch.zeros(freq_bins)
            sum_sq = torch.zeros(freq_bins)

        sum_ += xs.sum(dim=0)
        sum_sq += (xs**2).sum(dim=0)
        count += xs.shape[0]

    mean = sum_ / count
    var = sum_sq / count - mean**2
    std = torch.sqrt(torch.clamp(var, min=1e-8))

    return mean, std


def compute_per_freq_min_max(dataset):
    """
    Computes the min and max for each frequency in `dataset`.
    - `dataset`: the dataset to find the values for min and max in
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4
    )

    min_ = None
    max_ = None

    for batch in loader:
        # batch: (B, freq)
        xs = batch["x"]

        if min_ is None:
            freq_bins = xs.shape[1]
            min_ = torch.full((freq_bins,), float("inf"))
            max_ = torch.full((freq_bins,), float("-inf"))

        batch_min = xs.min(dim=0).values
        batch_max = xs.max(dim=0).values

        min_ = torch.minimum(min_, batch_min)
        max_ = torch.maximum(max_, batch_max)

    return min_, max_
