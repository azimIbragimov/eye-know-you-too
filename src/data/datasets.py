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
from torch.utils.data import Dataset

TASK_TO_NUM = {
    "HSS": 0,
    "RAN": 1,
    "TEX": 2,
    "FXS": 3,
    "VD1": 4,
    "VD2": 5,
    "BLG": 6,
}


class SubsequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: pd.DataFrame,
        subsequence_length: int,
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
                "exclude",
            )
        }
        for recording, (_, label) in zip(sequences, labels.iterrows()):
            nb_round = int(label["nb_round"])
            nb_subject = int(label["nb_subject"])
            nb_session = int(label["nb_session"])
            nb_task = TASK_TO_NUM[label["task"]]

            # Extract fixed-length, non-overlapping subsequences
            recording_tensor = torch.from_numpy(recording).float()
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
            metadata["exclude"].extend(exclude.numpy())

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
                    "exclude",
                )
            ]
        )

        return x, y
