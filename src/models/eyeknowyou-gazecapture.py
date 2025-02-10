# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

from typing import Iterable, Tuple
from pathlib import Path

from src.models.networks import Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import accuracy_calculator



class DuplicateLayer(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim  # Dimension along which to repeat

    def forward(self, x):
        return x.repeat_interleave(2, dim=self.dim)  # Duplicate each element

class Model(nn.Module):
    def __init__(
        self,
        n_classes: int,
        embeddings_filename: str,
        embeddings_dir: str = "./embeddings",
        w_metric_loss: float = 1.0,
        w_class_loss: float = 0.0,
        compute_map_at_r: bool = False,
        seq_len=5000,
    ):
        super().__init__()
        
        print("HERE", seq_len)

        input_shape: Iterable[int] = [9, 1024]
        self.normalize_embeddings: bool = False
        self.classifier = Classifier(128, n_classes)
        
        self.pad_method = self.pad_append

        self.input_shape = input_shape
        c_in, t_in = self.input_shape

        
        self.embedder = self.build_cnn(c_in, t_in, 
            {
            "n_layers": 7, 
            "filters": [64, 64, 64, 64, 128, 128, 128, 128, 256],
            "kernel_sizes": [3, 3, 3, 3, 3, 3, 3, 3, 3],
            "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256],
            "use_batch_norm": True
            })
        
        
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

        
    def __len__(self) -> int:
        """Count the number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def pad_append(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(
            x,
            (0, self.input_shape[-1] - x.shape[-1]),
            mode="constant",
            value=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = torch.repeat(x, 2)
        x = self.embedder(x)
        
        return x

    def build_cnn(
        self, c_in: int, t_in: int, params: dict
    ) -> Tuple[nn.Module, int, int]:
        n_layers: int = params["n_layers"]
        filters: Iterable[int] = params["filters"]
        kernel_sizes: Iterable[int] = params["kernel_sizes"]
        dilations: Iterable[int] = params["dilations"]
        use_batch_norm: bool = params["use_batch_norm"]

        layers = [DuplicateLayer(dim=-1)]
        for i in range(n_layers):
            c_out = filters[i]
            k = kernel_sizes[i]
            d = dilations[i]
            t_out = t_in - d * (k - 1)

            layers.append(nn.Conv1d(c_in, c_out, k, dilation=d))
            layers.append(nn.ReLU())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(c_out))

            c_in = c_out
            t_in = t_out
            
        layers.append(nn.Flatten())
        layers.append(
            self.build_fcn(44288, {
            "n_layers": 2,
            "sizes": [256, 128]
                                }
                           )
            )
        

        return nn.Sequential(*layers)
    
    def build_fcn(self, t_in: int, params: dict) -> Tuple[nn.Module, int]:
        n_layers: int = params["n_layers"]
        sizes: Iterable[int] = params["sizes"]
        use_batch_norm: bool = (
            False
            if "use_batch_norm" not in params
            else params["use_batch_norm"]
        )

        layers = []
        for i in range(n_layers):
            t_out = sizes[i]

            if i > 0:
                layers.append(nn.ReLU())
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(t_in))
            layers.append(nn.Linear(t_in, t_out))

            t_in = t_out

        return nn.Sequential(*layers)

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