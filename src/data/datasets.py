# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pathlib
import tqdm
from torch.utils.data import Dataset

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
    ):
        super().__init__()

        samples = []
        subjects = []
        metadata = {
            k: []
            for k in (
                "nb_round",
                "nb_subject",
                "nb_session",
                "nb_task",
                "nb_subsequence",
            )
        }

        for index, file in tqdm.tqdm(meta.iterrows(), total=meta.shape[0]):
            path = pathlib.Path(file["filename"])
            processed_path = processed_folder / f"{path.stem}.npy"
            
            nb_subject = int(file["part_id"])
            nb_round =int(file["round"])
            nb_session = int(file["session"])
            nb_task  = TASK_TO_NUM[file["task"]]
            
            if exclude_task == nb_task:
                continue

            # Extract fixed-length, non-overlapping subsequences
            data = np.load(processed_path).T
            recording_tensor = torch.from_numpy(data).float()
            subsequences = recording_tensor.unfold(
                dimension=-1, size=subsequence_length, step=subsequence_length
            )
            subsequences = subsequences.swapdims(0, 1)  # (batch, feature, seq)

            n_seq = subsequences.size(0)
            nb_subsequence = np.arange(n_seq)
            portion_nan = subsequences.isnan().any(dim=1).float().mean(dim=-1)
            exclude = portion_nan > 0.5

            samples.append(subsequences)
            subjects.append(torch.LongTensor([nb_subject] * n_seq))
            metadata["nb_round"].extend([nb_round] * n_seq)
            metadata["nb_subject"].extend([nb_subject] * n_seq)
            metadata["nb_session"].extend([nb_session] * n_seq)
            metadata["nb_task"].extend([nb_task] * n_seq)
            metadata["nb_subsequence"].extend(nb_subsequence)

        self.samples = torch.cat(samples, dim=0)
        self.metadata = pd.DataFrame(metadata)

        subjects = torch.cat(subjects, dim=0)
        unique_subjects = subjects.unique()
        self.classes = torch.bucketize(subjects, unique_subjects)
        self.n_classes = len(unique_subjects)

        if mn is None or sd is None:
            x = torch.clamp(self.samples, min=-1000.0, max=1000.0).numpy()
            mn = np.nanmean(x)
            sd = np.nanstd(x)
            
        self.mn = mn
        self.sd = sd

    def __len__(self) -> int:
        return self.samples.size(0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        x = self.samples[index]
        x = torch.clamp(x, min=-1000.0, max=1000.0)
        x = (x - self.mn) / self.sd
        x = torch.nan_to_num(x, nan=0.0)

        y = self.metadata.iloc[index, :].to_dict()
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

        return x, y