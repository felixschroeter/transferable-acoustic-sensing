"""
datasets.py

Contains the different types of datasets used in this work and a helper class for working with the audio files `AudioFile`,
i.e. the unlabeled and labeled domain datasets: `UnlabeledDomainDataset`, `LabeledDomainDataset`

"""

import tqdm
import math
import bisect
import torch

import preprocessing
import utils


class AudioFile:
    """
    A helper class to represent an audio sample.
    Provides classes for clean access and keeps the file in memory.
    """

    def __init__(self, path):
        """
        Initialize an audio file object that loads the audio file at `path`
        - `path`: the audio file to use
        """
        self.sound = preprocessing.load_audio(path)

    def __len__(self):
        return self.sound.shape[0]

    def __getitem__(self, i):
        """
        Returns a subset of entries from the audio file, supports slicing and adds zero padding if the slice is too large.
        - `i`: index or slice
        """
        if isinstance(i, slice):
            start, stop, _ = i.start, i.stop, i.step
            if stop <= len(self):
                sound = self.sound[start:stop]
            else:
                padding_length = stop - len(self)
                sound = torch.nn.functional.pad(self.sound[start:], (0, padding_length))
        else:
            sound = self.sound[i]

        return sound


class UnlabeledDomainDataset(torch.utils.data.Dataset):
    """
    An unlabeled domain dataset for samples recorded on one actuator.
    For multiple actuators use standard torch methods for combining datasets.
    """

    def __init__(
        self,
        paths,
        domain,
        window_size,
        stride,
        drop_before_sample=0,
        apply_transform=None,
        return_file_idx=False,
    ):
        """
        Creates a new UnlabeledDomainDataset.
        - `paths`: the file paths of all audio files to be included in this set
        - `domain`: the domain (i.e. actuator) that all the audio files were recorded on
        - `window_size`: how many entries from the file to retrieve at once,
            e.g. 48000 if we sample at 48000Hz and want one second samples
        - `stride`: offset from the start of one window to the next one
        - `drop_before_sample`: drop this many entries from the file before each window,
            can be used to remove breaks between sweeps
        - `apply_transform`: default None, supports "stft" or "fft" preprocessing of the data
        - `return_file_idx`: default False, returns the internal id for the audio file,
            should only be used if this dataset is wrapped in another dataset class such as LabeledDomainDataset
        """
        self.paths = paths
        self.files = [AudioFile(path) for path in paths]
        self.domain = torch.tensor(domain)

        self.return_file_idx = return_file_idx

        self.window_size = window_size
        self.stride = stride
        self.drop_before_sample = drop_before_sample

        max_index_per_file = [0]
        max_index = 0

        # for each of the files, collect how many windows fit inside and collect the max_id
        # of each audio file
        for file in tqdm.tqdm(self.files):
            file_length = len(file)

            max_index += (
                math.floor((file_length - window_size - drop_before_sample) // stride)
                + 1
            )
            max_index_per_file.append(max_index)

        self.max_index_per_file = max_index_per_file
        self.length = max_index_per_file[
            -1
        ]  # the max index of the last file corresponds to the length

        self.use_z_score_norm = False
        self.use_min_max_norm = False

        self.apply_transform = apply_transform

        if apply_transform == "stft":
            self.use_stft = True
            self.use_fft = False
        elif apply_transform == "fft":
            self.use_stft = False
            self.use_fft = True
        else:
            self.use_stft = False
            self.use_fft = False

    @classmethod
    def from_dataset_config(
        cls,
        domain,
        dataset_config,
        data_basedir,
        window_size,
        stride,
        drop_before_sample=0,
        apply_transform=None,
    ):
        """
        Create a new unlabeled dataset from a config.

        - `domain`: the domain (i.e. actuator) that all the audio files were recorded on
        - `dataset_config`: a hydra dataset config
        - `data_basedir`: the directory in which all folder for all datasets are saved
        - `window_size`: how many entries from the file to retrieve at once,
            e.g. 48000 if we sample at 48000Hz and want one second samples
        - `stride`: offset from the start of one window to the next one
        - `drop_before_sample`: drop this many entries from the file before each window,
            can be used to remove breaks between sweeps
        - `apply_transform`: default None, supports "stft" or "fft" preprocessing of the data

        """
        paths = utils.load_paths(dataset_config, data_basedir)
        return cls(
            paths,
            domain,
            window_size,
            stride,
            drop_before_sample,
            apply_transform=apply_transform,
        )

    def __len__(self):
        return self.length

    def _get_file_and_within_file_index(self, idx):
        """
        Find and return the internal file_idx where the sample with `idx` is stored.
        """
        file_id = bisect.bisect(self.max_index_per_file, idx) - 1

        within_file_idx = idx - self.max_index_per_file[file_id]
        return file_id, within_file_idx

    def __getitem__(self, idx):
        """
        Retrieve the sample `idx` from this dataset.
        returns a dict of:
            - `x`: the sample
            - `domain`: the domain of the sample
            - `label`: a regression or classification label, (dummy value for this class)
            - `has_label`: always false for this dataset
        """
        # find out which file this id is saved in
        file_id, in_file_idx = self._get_file_and_within_file_index(idx)

        # get the offset inside the file
        start = self.drop_before_sample + in_file_idx * self.stride

        # retrieve the sample
        x = self.files[file_id][start : start + self.window_size]

        # apply preprocessing
        if self.use_stft:
            x = preprocessing.apply_stft(x)
        elif self.use_fft:
            x = preprocessing.apply_fft(x)

        # apply norm
        if self.use_z_score_norm:
            x = (x - self.freq_mean) / (self.freq_std + 1e-8)
        elif self.use_min_max_norm:
            x = (x - self.freq_min) / (self.freq_max - self.freq_min)

        if self.return_file_idx:
            return x, self.domain, file_id
        else:
            out = {
                "x": x,
                "domain": self.domain,
                "label": torch.tensor(0).float(),  # dummy label
                "has_label": False,  # no samples from this dataset have a label
            }
            return out

    def set_z_score_norm_params(self, freq_mean, freq_std):
        """
        Set the norm parameters for the z-score norm
        """
        self.use_z_score_norm = True
        self.freq_mean = freq_mean
        self.freq_std = freq_std

    def set_min_max_norm_params(self, freq_min, freq_max):
        """
        Set the norm parameters for the min-max norm
        """
        self.use_min_max_norm = True
        self.freq_min = freq_min
        self.freq_max = freq_max


# mostly use for domain labels, or one file -> one embedding
class LabeledDomainDataset(torch.utils.data.Dataset):
    """
    Thin wrapper around the UnlabeledDomainDataset that only administers the labels.
    """

    def __init__(self, unlabeled_domain_dataset, file_labels):
        """
        Create a new LabeledDomainDataset from an UnlabeledDomainDataset and file labels
        - `unlabeled_domain_dataset`: the dataset
        - `file_labels`: the labels
        """
        assert len(unlabeled_domain_dataset) == len(file_labels)
        assert unlabeled_domain_dataset.return_file_idx

        self.unlabeled_domain_dataset = unlabeled_domain_dataset
        self.file_labels = file_labels

    @classmethod
    def from_dataset_config(
        cls,
        domain,
        dataset_config,
        data_basedir,
        window_size,
        stride,
        drop_before_sample=0,
        apply_transform=None,
        is_regression=True,
        label_dict=utils.LABEL_DICT_CLASSIFICATION,
    ):
        """
        Create a new labeled dataset from a config.

        - `domain`: the domain (i.e. actuator) that all the audio files were recorded on
        - `dataset_config`: a hydra dataset config
        - `data_basedir`: the directory in which all folder for all datasets are saved
        - `window_size`: how many entries from the file to retrieve at once,
            e.g. 48000 if we sample at 48000Hz and want one second samples
        - `stride`: offset from the start of one window to the next one
        - `drop_before_sample`: drop this many entries from the file before each window,
            can be used to remove breaks between sweeps
        - `apply_transform`: default None, supports "stft" or "fft" preprocessing of the data
        - `is_regression`: if the labels are regression or classification labels
        - `label_dict`: dict to map classification strings to a numerical representation
        """
        paths, labels = utils.load_paths_and_labels(dataset_config, data_basedir)

        if is_regression:
            # turn the "no touch" regression label into -1
            labels = [float(label) if label != "none" else -1.0 for label in labels]
        else:
            labels = [label_dict[label] for label in labels]

        # create the unlabeled dataset
        unlabeled_domain_dataset = UnlabeledDomainDataset(
            paths,
            domain,
            window_size,
            stride,
            drop_before_sample,
            apply_transform=apply_transform,
            return_file_idx=True,
        )
        file_labels = torch.tensor(labels)

        return cls(
            unlabeled_domain_dataset=unlabeled_domain_dataset,
            file_labels=file_labels,
        )

    def __len__(self):
        return len(self.unlabeled_domain_dataset)

    def __getitem__(self, idx):
        """
        Retrieve the sample `idx` from this dataset.
        returns a dict of:
            - `x`: the sample
            - `domain`: the domain of the sample
            - `label`: a regression or classification label, (dummy value for this class)
            - `has_label`: always false for this dataset
        """
        x, domain, file_id = self.unlabeled_domain_dataset.__getitem__(idx)
        label = self.file_labels[file_id]
        out = {"x": x, "domain": domain, "label": label, "has_label": True}
        return out

    def set_z_score_norm_params(self, freq_mean, freq_std):
        """
        Set the norm parameters for the z-score norm
        """
        return self.unlabeled_domain_dataset.set_z_score_norm_params(
            freq_mean=freq_mean, freq_std=freq_std
        )

    def set_min_max_norm_params(self, freq_min, freq_max):
        """
        Set the norm parameters for the min-max norm
        """
        return self.unlabeled_domain_dataset.set_min_max_norm_params(
            freq_min=freq_min, freq_max=freq_max
        )
