import pickle
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Sequence
import torch

import numpy as np
import pandas as pd

from pytorch_metric_learning import samplers
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..assign_groups import assign_groups
from .dataloader import SubsequenceDataset
from ..download import download
from ..downsample import downsample_recording
from ..zip import extract
from ..base_dataset import BaseDataset

class Dataset(BaseDataset):
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
        num_workers: int = 1,
        cache_size: int = 8e9
    ):
        super().__init__()
        
        self.TASK_TO_NUM = {}
        self.train_exclude_task = None

        self.initial_sampling_rate_hz = 1000
        self.downsample_factors = downsample_factors
        self.total_downsample_factor = np.prod(self.downsample_factors)
        self.noise_sd = noise_sd
        self.num_workers = num_workers
        self.cache_size = cache_size
        
        self.subsequence_length = int(
            subsequence_length_before_downsampling
            // self.total_downsample_factor
        )

        self.base_dir = Path(base_dir)
        self.processed_file_dir = self.base_dir / "processed"
        

        self.current_fold = current_fold

        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.fit_batch_size = self.classes_per_batch * self.samples_per_class

        self.train_loader: DataLoader
        self.val_loaders: List[DataLoader]
        self.test_loaders: List[DataLoader]

        self.n_classes: int
        self.zscore_mn: float
        self.zscore_sd: float

    def setup(self, stage: Optional[str] = None) -> None:        
        # Get metadata from train, val and test sets        
        metadata = pd.read_csv(self.processed_file_dir / "metadata.csv")
        train_metadata = metadata[(metadata["set"] != -1) & (metadata["set"] != self.current_fold)]        
        val_metadata = metadata[metadata["set"] == self.current_fold]
        test_metadata = metadata[metadata["set"] == -1]
        
                
        train_set = SubsequenceDataset(
            train_metadata, 
            self.subsequence_length, 
            self.TASK_TO_NUM, 
            self.train_exclude_task,
            self.processed_file_dir,
            mn=None, 
            sd=None,
            cache_size=self.cache_size
        )
        
        self.zscore_mn = train_set.mn
        self.zscore_sd = train_set.sd
        self.n_classes = train_set.n_classes
        
        train_sampler = samplers.MPerClassSampler(
            train_set.classes,
            self.samples_per_class,
            batch_size=self.fit_batch_size,
            length_before_new_iter=len(train_set),
        )
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.fit_batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
        )

        full_val_set = SubsequenceDataset(
            val_metadata,
            self.subsequence_length,
            self.TASK_TO_NUM,
            self.train_exclude_task,
            self.processed_file_dir,
            mn=self.zscore_mn,
            sd=self.zscore_sd,
            cache_size=self.cache_size
        )
       
        full_val_sampler = samplers.MPerClassSampler(
                full_val_set.classes,
                self.samples_per_class,
                batch_size=self.fit_batch_size,
                length_before_new_iter=len(full_val_set),
            )
        
        self.full_val_loader = DataLoader(
            full_val_set,
            batch_size=self.fit_batch_size,
            shuffle=False,
            sampler=full_val_sampler,
            num_workers=self.num_workers,
        )

        test_set = SubsequenceDataset(
                test_metadata,
                self.subsequence_length,
                self.TASK_TO_NUM,
                None,
                self.processed_file_dir,
                mn=self.zscore_mn,
                sd=self.zscore_sd,
                cache_size=self.cache_size
            )
        
        self.test_loader = DataLoader(
                test_set,
                batch_size=self.fit_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.full_val_loader

    def test_dataloader(self) -> DataLoader:
        return self.test_loader