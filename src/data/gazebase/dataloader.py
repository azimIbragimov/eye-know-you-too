from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import pathlib
import tqdm
from torch.utils.data import Dataset
from functools import lru_cache
import functools
import os

class SubsequenceDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        subsequence_length: int,
        TASK_TO_NUM: dict,
        exclude_task: str,
        processed_folder: str,
        mn: Optional[float] = None,
        sd: Optional[float] = None,
        cache_size: int = int(1e9) 
    ):
        super().__init__()
        self.subsequence_length = subsequence_length
        self.processed_folder = processed_folder
        self.TASK_TO_NUM = TASK_TO_NUM
        self.exclude_task = exclude_task
        self.cache_size = cache_size

        global load_cached_npy
        @functools.lru_cache(maxsize=int(self.cache_size / 1e6))
        def load_cached_npy(file_path: str):
            return np.load(file_path).T

        self.load_cached_npy = load_cached_npy
        
        subjects = []
        metadata = {
            k: []
            for k in (
                "path",
                "nb_round",
                "nb_subject",
                "nb_session",
                "nb_task",
                "nb_subsequence",
            )
        }

        n, mn_temp, std_temp = 0, 0, 0
        
        for index, file in tqdm.tqdm(meta.iterrows(), total=meta.shape[0]):
            path = pathlib.Path(file["filename"])
            processed_path = self.processed_folder / f"{path.stem}.npy"
            
            nb_subject = int(file["part_id"])
            nb_round = int(file["round"])
            nb_session = int(file["session"])
            nb_task = self.TASK_TO_NUM[file["task"]]
            
            if self.exclude_task == nb_task:
                continue

            # Load data with LRU cache
            data = self.load_cached_npy(str(processed_path))
            recording_tensor = torch.from_numpy(data).float()
            subsequences = recording_tensor.unfold(
                dimension=-1, size=self.subsequence_length, step=self.subsequence_length
            ).swapdims(0, 1)  # (batch, feature, seq)
            
            n += 1
            mn_temp += np.nanmean(subsequences)
            std_temp += np.nanstd(subsequences)

            n_seq = subsequences.size(0)
            nb_subsequence = np.arange(n_seq)
            portion_nan = subsequences.isnan().any(dim=1).float().mean(dim=-1)
            exclude = portion_nan > 0.5

            subjects.append(torch.LongTensor([nb_subject] * n_seq))
            metadata["path"].extend([processed_path] * n_seq)
            metadata["nb_round"].extend([nb_round] * n_seq)
            metadata["nb_subject"].extend([nb_subject] * n_seq)
            metadata["nb_session"].extend([nb_session] * n_seq)
            metadata["nb_task"].extend([nb_task] * n_seq)
            metadata["nb_subsequence"].extend(nb_subsequence)

        self.metadata = pd.DataFrame(metadata)
        subjects = torch.cat(subjects, dim=0)
        unique_subjects = subjects.unique()
        self.classes = torch.bucketize(subjects, unique_subjects)
        self.n_classes = len(unique_subjects)
        
        if mn is None or sd is None:
            self.mn = mn_temp / n
            self.sd = std_temp / n
        else:        
            self.mn = mn
            self.sd = sd

    def __len__(self) -> int:
        return self.metadata.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        y = self.metadata.iloc[index, :].to_dict()
        path = y["path"]
        y["class"] = self.classes[index]
        y = torch.LongTensor(
            [
                y[k]
                for k in (
                    "class",
                    "nb_round",
                    "nb_subject",
                    "nb_session",
                    "nb_task",
                    "nb_subsequence",
                )
            ]
        )

        # Load data with LRU cache
        x = self.load_cached_npy(str(path))
        x = torch.from_numpy(x).float()
        x = x.unfold(dimension=-1, size=self.subsequence_length, step=self.subsequence_length).swapdims(0,1)
        x = x[y[-1]]
        
        x = torch.clamp(x, min=-1000.0, max=1000.0)
        x = (x - self.mn) / self.sd
        x = torch.nan_to_num(x, nan=0.0)
        return x, y