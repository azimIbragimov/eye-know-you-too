import pickle
import re
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from pytorch_metric_learning import samplers
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

from .assign_groups import assign_groups
from .datasets import SubsequenceDataset
from .download import download
from .downsample import downsample_recording
from .zip import extract

# https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257/3
GAZEBASE_URL = "https://ndownloader.figshare.com/files/27039812"

# https://osf.io/5zpvk/
JUDO1000_URL = "https://osf.io/4wy7s/download"


class GazeBase():
    def __init__(
        self,
        current_fold: int = 0,
        base_dir: str = "./data/gazebase_v3",
        downsample_factors: Sequence[int] = tuple(),
        subsequence_length_before_downsampling: int = 5000,
        classes_per_batch: int = 16,
        samples_per_class: int = 16,
        compute_map_at_r: bool = False,
        batch_size_for_testing: Optional[int] = None,
        noise_sd: Optional[float] = None,
    ):
        super().__init__()

        self.initial_sampling_rate_hz = 1000
        self.downsample_factors = downsample_factors
        self.total_downsample_factor = np.prod(self.downsample_factors)
        self.noise_sd = noise_sd

        self.subsequence_length = int(
            subsequence_length_before_downsampling
            // self.total_downsample_factor
        )

        self.base_dir = Path(base_dir)
        self.archive_path = self.base_dir / "gazebase.zip"
        self.raw_file_dir = self.base_dir / "raw"
        self.tmp_file_dir = self.base_dir / "tmp"
        self.processed_path = (
            self.base_dir
            / "processed"
            / (
                f"gazebase_savgol_ds{int(self.total_downsample_factor)}"
                + f"_{'normal' if self.noise_sd is None else 'degraded'}.pkl"
            )
        )

        self.current_fold = current_fold
        self.n_folds = 4
        self.nb_round_for_test_subjects = 6

        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.fit_batch_size = self.classes_per_batch * self.samples_per_class
        self.test_batch_size = batch_size_for_testing or self.fit_batch_size

        self.compute_map_at_r = compute_map_at_r

        self.train_loader: DataLoader
        self.val_loaders: List[DataLoader]
        self.test_loaders: List[DataLoader]

        self.n_classes: int
        self.zscore_mn: float
        self.zscore_sd: float

    def prepare_data(self) -> None:
        if not self.processed_path.exists():
            self.download_and_process_gazebase()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "test":
            batch_size = self.test_batch_size
        else:
            batch_size = self.fit_batch_size

        fold_label = f"fold{self.current_fold}"
        with open(self.processed_path, "rb") as f:
            data_dict = pickle.load(f)

        print(len(data_dict["fold0"]["inputs"][0]))

        train_X = [
            x.T  # shape: (feature, seq)
            for split, v in data_dict.items()
            if split not in (fold_label, "test")
            for x in v["inputs"]
        ]
        train_y = pd.concat(
            [
                v["labels"]
                for split, v in data_dict.items()
                if split not in (fold_label, "test")
            ],
            ignore_index=True,
            axis=0,
        )

        # Remove BLG data from train set
        is_train_blg = train_y["task"].str.fullmatch("BLG")
        train_X = [x for x, is_blg in zip(train_X, is_train_blg) if not is_blg]
        train_y = train_y.loc[~is_train_blg, :]

        train_set = SubsequenceDataset(
            train_X, train_y, self.subsequence_length, mn=None, sd=None
        )
        self.zscore_mn = train_set.mn
        self.zscore_sd = train_set.sd
        self.n_classes = train_set.n_classes
        train_sampler = samplers.MPerClassSampler(
            train_set.classes,
            self.samples_per_class,
            batch_size=batch_size,
            length_before_new_iter=len(train_set),
        )
        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=1,
        )

        # We want multiple "validation sets" for different purposes:
        # 1. Full val set for comparing val loss to train loss (or, at
        #    test time, for computing all val-set embeddings)
        # 2. TEX-only val set for measuring MAP@R
        # 3. TEX-only train set for measuring MAP@R
        # If we don't want to measure MAP@R, (2) and (3) can be omitted.
        val_X = [x.T for x in data_dict[fold_label]["inputs"]]
        val_y = data_dict[fold_label]["labels"]
        self.val_loaders = []

        if stage != "test":
            # Remove BLG data from val set when training/tuning
            is_val_blg = val_y["task"].str.fullmatch("BLG")
            val_X = [x for x, is_blg in zip(val_X, is_val_blg) if not is_blg]
            val_y = val_y.loc[~is_val_blg, :]

        full_val_set = SubsequenceDataset(
            val_X,
            val_y,
            self.subsequence_length,
            mn=self.zscore_mn,
            sd=self.zscore_sd,
        )
        if stage == "test":
            # When testing, we want to embed all data samples
            full_val_sampler = None
        else:
            # When fitting, we want to compute multi-similarity loss
            full_val_sampler = samplers.MPerClassSampler(
                full_val_set.classes,
                self.samples_per_class,
                batch_size=batch_size,
                length_before_new_iter=len(full_val_set),
            )
        full_val_loader = DataLoader(
            full_val_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=full_val_sampler,
            num_workers=1,
        )
        self.val_loaders.append(full_val_loader)

        if stage != "test" and self.compute_map_at_r:
            val_is_tex = val_y["task"].str.fullmatch("TEX")
            val_tex_X = [x for x, is_tex in zip(val_X, val_is_tex) if is_tex]
            val_tex_y = val_y.loc[val_is_tex, :]
            val_tex_set = SubsequenceDataset(
                val_tex_X,
                val_tex_y,
                self.subsequence_length,
                mn=self.zscore_mn,
                sd=self.zscore_sd,
            )
            val_tex_loader = DataLoader(
                val_tex_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
            )
            self.val_loaders.append(val_tex_loader)

            train_is_tex = train_y["task"].str.fullmatch("TEX")
            train_tex_X = [
                x for x, is_tex in zip(train_X, train_is_tex) if is_tex
            ]
            train_tex_y = train_y.loc[train_is_tex, :]
            train_tex_set = SubsequenceDataset(
                train_tex_X,
                train_tex_y,
                self.subsequence_length,
                mn=self.zscore_mn,
                sd=self.zscore_sd,
            )
            train_tex_loader = DataLoader(
                train_tex_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
            )
            self.val_loaders.append(train_tex_loader)

        self.test_loaders = []
        if stage == "test":
            test_X = [x.T for x in data_dict["test"]["inputs"]]
            test_y = data_dict["test"]["labels"]
            test_set = SubsequenceDataset(
                test_X,
                test_y,
                self.subsequence_length,
                mn=self.zscore_mn,
                sd=self.zscore_sd,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=1,
            )
            self.test_loaders.append(test_loader)
            self.test_loaders.append(full_val_loader)

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loaders

    def test_dataloader(self) -> DataLoader:
        return self.test_loaders

    def download_and_process_gazebase(self) -> None:
        # Download and extract GazeBase archives if necessary
        if not self.raw_file_dir.exists() or self.tmp_file_dir.exists():
            self.archive_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.archive_path.exists():
                print("Downloading GazeBase from figshare")
                download(GAZEBASE_URL, self.archive_path)

            # If the temporary directory still exists, there must have
            # been an error when extracting files.  Delete the old
            # directories to start fresh.
            if self.tmp_file_dir.exists():
                print("Removing old directories")
                shutil.rmtree(self.tmp_file_dir)
                if self.raw_file_dir.exists():
                    shutil.rmtree(self.raw_file_dir)

            print("Extracting GazeBase archive to temporary directory")
            self.tmp_file_dir.mkdir(parents=True, exist_ok=True)
            extract(self.archive_path, self.tmp_file_dir)

            print("Extracting subject archives to temporary directory")
            subject_archives = list(self.tmp_file_dir.rglob("Subject_*.zip"))
            for archive in tqdm(subject_archives):
                extract(archive, self.tmp_file_dir)

            print("Moving data files out of temporary directory")
            self.raw_file_dir.mkdir(parents=True, exist_ok=True)
            data_files = list(self.tmp_file_dir.rglob("S_*.csv"))
            for file in tqdm(data_files):
                new_file_path = self.raw_file_dir / file.name
                shutil.move(file, new_file_path)

            print("Deleting temporary directory")
            shutil.rmtree(self.tmp_file_dir)

        # Process recordings
        filename_pattern = r"S_(\d)(\d+)_S(\d)_(\w+)"
        recording_paths = sorted(list(self.raw_file_dir.iterdir()))
        inputs = []
        labels = []
        print("Processing all recordings")
        for path in tqdm(recording_paths):
            df = pd.read_csv(path)
            gaze, ideal_sampling_rate = downsample_recording(
                df, self.downsample_factors, self.initial_sampling_rate_hz
            )
            if self.noise_sd is not None:
                noise = np.random.randn(*gaze.shape) * self.noise_sd
                gaze += noise
            vel = savgol_filter(gaze, 7, 2, deriv=1, axis=0, mode="nearest")
            vel *= ideal_sampling_rate  # deg/sec

            inputs.append(vel.astype(np.float32))


            pattern_match = re.match(filename_pattern, path.stem)
            match_groups = pattern_match.groups()
            label = {
                "nb_round": match_groups[0],
                "nb_subject": match_groups[1],
                "nb_session": match_groups[2],
                "task": match_groups[3],
            }
            labels.append(label)
        labels_df = pd.DataFrame(labels)

        # The test set contains all the data from subjects who are
        # present in Round 6.  This comprises 59 subjects and 49.26% of
        # all recordings in GazeBase.
        print("Creating held-out test set")
        subjects = labels_df.loc[:, "nb_subject"]
        nb_round_as_int = labels_df.loc[:, "nb_round"].astype(int)
        is_test_round = nb_round_as_int == self.nb_round_for_test_subjects
        subjects_in_test_set = subjects[is_test_round].unique()
        is_subject_in_test_set = subjects.isin(subjects_in_test_set)

        n_subjects = len(subjects.unique())
        test_pct_subjects = 100 * len(subjects_in_test_set) / n_subjects
        test_pct_recordings = 100 * is_subject_in_test_set.mean()
        print(
            f"Created test set with {test_pct_subjects:.2f}% of subjects"
            + f", {test_pct_recordings:.2f}% of recordings"
        )

        print("Assigning remaining subjects to folds")
        leftover_subjects = subjects[~is_subject_in_test_set]
        leftover_unique = leftover_subjects.unique()

        # Weight each subject by the number of recordings they have.
        # Since heapq prioritizes smaller weights but we want to
        # prioritize higher weights, we negate our weights to prioritize
        # them correctly.
        weights = [-np.sum(leftover_subjects == s) for s in leftover_unique]
        fold_to_id, grp = assign_groups(self.n_folds, leftover_unique, weights)

        # Verify that the folds are roughly balanced
        least, greatest = np.min(grp, axis=0), np.max(grp, axis=0)
        subject_diff = greatest[0] - least[0]
        recording_diff = greatest[1] - least[1]
        print(f"Max - min of # subjects in each fold: {subject_diff}")
        print(f"Max - min of # recordings in each fold: {recording_diff}")

        # Create a dictionary of inputs and labels for each data split
        def get_split_data(split_subjects):
            split_indices = np.where(subjects.isin(split_subjects))[0]
            return {
                "inputs": [inputs[i] for i in split_indices],
                "labels": labels_df.iloc[split_indices, :],
            }

        test_idx = {"test": get_split_data(subjects_in_test_set)}
        fold_idx = {
            f"fold{k}": get_split_data(v) for k, v in fold_to_id.items()
        }
        data_dict = {**test_idx, **fold_idx}

        # Save dictionary of processed data
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.processed_path, "wb") as f:
            pickle.dump(data_dict, f)
        print(f"Finished processing data. Saved to '{self.processed_path}'.")


