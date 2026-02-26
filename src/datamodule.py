"""
datamodule.py

Contains the code for our AudioDatamodule which encapsulates the code for creating datasets and dataloaders,
given a set of configurations.

This is mostly used directly by torch lightning that natively interacts with it.

"""

import lightning
import torch

from torch.utils.data import ConcatDataset

from datasets import (
    LabeledDomainDataset,
    UnlabeledDomainDataset,
)

import utils


class AudioDataModule(lightning.LightningDataModule):
    """
    Encapsulates the logic for creating datasets and dataloaders from the configs.
    Works natively with torch lighnting and is called from the lightning module directly.

    """

    def __init__(
        self,
        model_name,
        data_basedir,
        train_datasets,
        train_regression_datasets,
        test_datasets,
        test_classification_datasets,
        test_regression_datasets,
        datasets,
        window_size,
        stride,
        apply_transform,
        test_size,
        batch_size,
        num_workers,
        persistent_workers,
        prefetch_factor,
        pin_memory,
        random_seed,
        apply_norm="z-score",
    ):
        """
        Initialize a new AudioDataModule with the corresponding config parameters.
        The parametres are explained in their corresponding hydra configs.
        """
        super().__init__()

        self.model_name = model_name

        # base directory where all recordings are sctores
        self.data_basedir = data_basedir

        # dataset configs
        self.datasets = datasets

        # dataset ids for training and testing
        self.train_datasets = train_datasets
        self.train_regression_datasets = train_regression_datasets
        self.test_datasets = test_datasets
        self.test_classification_datasets = test_classification_datasets
        self.test_regression_datasets = test_regression_datasets

        # dataset and preprocessing parameters
        self.window_size = window_size
        self.stride = stride
        self.apply_transform = apply_transform
        self.apply_norm = apply_norm

        # random seed and test_size
        self.random_seed = random_seed
        self.test_size = test_size

        # dataloading parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        # helper mappings
        self.finger_id_to_domain = {}
        self.finger_id_to_norm_params = {}

    def get_configs(self, dataset_ids):
        """
        Return all config objects for all ids in `dataset_ids`
        """
        return [self.datasets[dataset_id] for dataset_id in dataset_ids]

    def dataset_from_config(self, config, is_unlabeled=False, is_regression=True):
        """
        Create a dataset from a dataset config.
        - `config`: the configuration, a hydra config for the dataset as defined in src/conf/datasets
        - `is_unlabeled`: If true, creates an unlabeled dataset else expects labels in the filenames
        - `is_regression`: Indiciates if the dataset has regression or classification labels

        returns a Dataset that corresponds to the config and parameters
        """
        # pick the right dataset class
        class_ = UnlabeledDomainDataset if is_unlabeled else LabeledDomainDataset

        # add this finger to the domain mapping if it was not assigned an integer for the domain already
        if config["finger_id"] not in self.finger_id_to_domain.keys():
            self.finger_id_to_domain = utils.get_finger_id_domain_mapping(
                [config], self.finger_id_to_domain
            )

        # create arguments for creating the dataset
        kwargs = {
            "domain": self.finger_id_to_domain[config["finger_id"]],
            "dataset_config": config,
            "data_basedir": self.data_basedir,
            "window_size": self.window_size,
            "stride": self.stride,
            "apply_transform": self.apply_transform,
        }
        if not is_unlabeled:
            kwargs["is_regression"] = is_regression

        return class_.from_dataset_config(**kwargs)

    def setup(self, stage):
        """
        Create the datasets for the `stage` that it is called with.
        Loads the data, preprocesses and normalized it and then sets the datasets as class attributes.

        - `stage`: "fit" for the training stage and "test" for the test stage

        """

        if stage == "fit":
            # load the configs
            train_unlabeled_configs = self.get_configs(self.train_datasets)
            train_regression_configs = self.get_configs(self.train_regression_datasets)

            # create the finger_id to integer domain mappings
            self.finger_id_to_domain = utils.get_finger_id_domain_mapping(
                train_unlabeled_configs + train_regression_configs
            )

            # create mapping of finger_id to all related datasets
            finger_id_to_datasets = {k: [] for k in self.finger_id_to_domain}

            # create the datasets
            for config in train_unlabeled_configs:
                finger_id = config["finger_id"]
                dataset = self.dataset_from_config(config=config, is_unlabeled=True)

                finger_id_to_datasets[finger_id].append(dataset)
            for config in train_regression_configs:
                finger_id = config["finger_id"]
                dataset = self.dataset_from_config(config=config)

                finger_id_to_datasets[finger_id].append(dataset)

            # train/val splits and normalization
            # norm on all train for a finger id, reapply to all val for a finger id, save for test
            self.finger_id_to_norm_params = {}

            train_datasets_unlabeled = []
            val_datasets_unlabeled = []

            train_datasets_regression = []
            val_datasets_regression = []

            generator = torch.Generator().manual_seed(self.random_seed)
            for finger_id, datasets in finger_id_to_datasets.items():
                use_for_norm = []
                all_datasets_this_finger = []
                for dataset in datasets:
                    train_dataset, val_dataset = torch.utils.data.random_split(
                        dataset,
                        [1 - self.test_size, self.test_size],
                        generator=generator,
                    )

                    if isinstance(dataset, UnlabeledDomainDataset):
                        train_datasets_unlabeled.append(train_dataset)
                        val_datasets_unlabeled.append(val_dataset)
                    elif isinstance(dataset, LabeledDomainDataset):
                        train_datasets_regression.append(train_dataset)
                        val_datasets_regression.append(val_dataset)
                    else:
                        raise NotImplementedError()

                    use_for_norm.append(train_dataset)
                    all_datasets_this_finger.append(dataset)

                if self.apply_norm == "z-score":
                    mean, std = utils.compute_per_freq_mean_std(
                        ConcatDataset(use_for_norm)
                    )
                    self.finger_id_to_norm_params[finger_id] = (mean, std)
                    for dataset in all_datasets_this_finger:
                        dataset.set_z_score_norm_params(mean, std)

                elif self.apply_norm == "min-max":
                    min, max = utils.compute_per_freq_min_max(
                        ConcatDataset(use_for_norm)
                    )
                    self.finger_id_to_norm_params[finger_id] = (min, max)
                    for dataset in all_datasets_this_finger:
                        dataset.set_min_max_norm_params(min, max)
            self.train_dataset, self.val_dataset = (
                train_datasets_unlabeled,
                val_datasets_unlabeled,
            )

            self.train_regression_dataset, self.val_regression_dataset = (
                train_datasets_regression,
                val_datasets_regression,
            )

        if stage == "test":
            configs_classification = self.get_configs(self.test_classification_datasets)
            configs_regression = self.get_configs(self.test_regression_datasets)

            test_regression_datasets = []
            finger_ids_regression = []
            for config in configs_regression:
                finger_id = config["finger_id"]
                finger_ids_regression.append(finger_id)

                dataset = self.dataset_from_config(config=config)

                if self.apply_norm == "z-score":
                    if finger_id in self.finger_id_to_norm_params.keys():
                        mean, std = self.finger_id_to_norm_params[finger_id]
                    else:
                        mean, std = utils.compute_per_freq_mean_std(dataset)
                    dataset.set_z_score_norm_params(mean, std)

                elif self.apply_norm == "min-max":
                    if finger_id in self.finger_id_to_norm_params.keys():
                        min, max = self.finger_id_to_norm_params[finger_id]
                    else:
                        min, max = utils.compute_per_freq_min_max(dataset)
                    dataset.set_min_max_norm_params(min, max)
                test_regression_datasets.append(dataset)

            test_classification_datasets = []
            finger_ids_classification = []
            for config in configs_classification:
                finger_id = config["finger_id"]
                finger_ids_classification.append(finger_id)

                dataset = self.dataset_from_config(config=config, is_regression=False)

                if self.apply_norm == "z-score":
                    if finger_id in self.finger_id_to_norm_params.keys():
                        mean, std = self.finger_id_to_norm_params[finger_id]
                    else:
                        mean, std = utils.compute_per_freq_mean_std(dataset)
                    dataset.set_z_score_norm_params(mean, std)

                elif self.apply_norm == "min-max":
                    if finger_id in self.finger_id_to_norm_params.keys():
                        min, max = self.finger_id_to_norm_params[finger_id]
                    else:
                        min, max = utils.compute_per_freq_min_max(dataset)
                    dataset.set_min_max_norm_params(min, max)
                test_classification_datasets.append(dataset)
            self.test_datasets = test_classification_datasets + test_regression_datasets
            self.test_dataset_finger_idx = (
                finger_ids_classification + finger_ids_regression
            )

            # create idx lists to map dataloader ids back to task (classification/regression)
            dataset_count_classification = len(configs_classification)
            dataset_count_regression = len(configs_regression)

            self.test_classification_dataloader_idx = range(
                0, dataset_count_classification
            )
            self.test_regression_dataloader_idx = range(
                dataset_count_classification,
                dataset_count_classification + dataset_count_regression,
            )

    def train_dataloader(self):
        """
        Create and return the train dataloaders
        """
        datasets = ConcatDataset(self.train_dataset + self.train_regression_dataset)

        return torch.utils.data.DataLoader(
            datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Create and return the validation dataloaders
        """
        datasets = [self.val_dataset, self.val_regression_dataset]
        datasets = ConcatDataset(self.val_dataset + self.val_regression_dataset)

        return torch.utils.data.DataLoader(
            datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        """
        Create and return the test dataloaders.
        """
        dataloaders = [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
            )
            for dataset in self.test_datasets
        ]
        return dataloaders
