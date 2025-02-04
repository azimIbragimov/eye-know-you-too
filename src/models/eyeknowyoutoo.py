from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import accuracy_calculator

from src.models.networks import Classifier, SimpleDenseNet
from src.models.base_model import BaseModel

class Model(BaseModel):
    def __init__(
        self,
        n_classes: int,
        embeddings_filename: str,
        embeddings_dir: str = "./embeddings",
        w_metric_loss: float = 1.0,
        w_class_loss: float = 0.1,
        compute_map_at_r: bool = False,
    ):
        super().__init__()

        self.embedder = SimpleDenseNet(9, 128)

        self.classifier = Classifier(128, n_classes)

        self.w_metric_loss = w_metric_loss
        self.metric_criterion = losses.MultiSimilarityLoss()
        self.metric_miner = miners.MultiSimilarityMiner()

        self.w_class_loss = w_class_loss
        self.class_criterion = torch.nn.CrossEntropyLoss()

        self.compute_map_at_r = compute_map_at_r
        self.map_at_r_calculator = (
            accuracy_calculator.AccuracyCalculator(
                include=["mean_average_precision_at_r"],
                avg_of_avgs=True,
                k="max_bin_count",
            )
            if self.compute_map_at_r
            else None
        )

        self.embeddings_path = Path(embeddings_dir) / embeddings_filename

    def forward(self, x):
        out = self.embedder(x)
        return out


    def metric_step(
        self, embeddings: torch.Tensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        if self.w_metric_loss <= 0.0:
            return 0.0

        mined_indices = (
            None
            if self.metric_miner is None
            else self.metric_miner(embeddings, labels)
        )
        metric_loss = self.metric_criterion(embeddings, labels, mined_indices)

        weighted_metric_loss = metric_loss * self.w_metric_loss
        return weighted_metric_loss

    def class_step(
        self, embeddings: torch.Tensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        # Since we have class-disjoint datasets, only compute class loss
        # on the training set.  We know we're working with the train set
        # if `self.embedder.training` is True.
        if (
            self.classifier is None
            or self.w_class_loss <= 0.0
            or not self.training
        ):
            return 0.0

        # When logits and labels are on the GPU, we get an error on the
        # backward pass.  For some reason, transferring them to the CPU
        # fixes the error.
        #
        # Full error below (with several instances of this error for
        # different threads, e.g., [6,0,0], [7,0,0], and [13,0,0]):
        # .../pytorch_1634272168290/work/aten/src/ATen/native/cuda/Loss.cu:455:
        # nll_loss_backward_reduce_cuda_kernel_2d: block: [0,0,0],
        # thread: [5,0,0] Assertion `t >= 0 && t < n_classes` failed.
        logits = self.classifier(embeddings)
        class_loss = self.class_criterion(logits.cpu(), labels.cpu())

        weighted_class_loss = class_loss * self.w_class_loss
        return weighted_class_loss

