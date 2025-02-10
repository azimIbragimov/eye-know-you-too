# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

from typing import Iterable, Tuple, List
from pathlib import Path

from src.models.networks import Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import accuracy_calculator


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """1D Convolutional Block: Conv1D + Activation + BatchNorm + Pooling"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', pool_size=2, activation='relu'):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=True
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()
        self.pool = nn.AvgPool1d(pool_size, stride=2, padding=1)  # AvgPool1D with 'same' padding

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        x = self.pool(x)
        return x

class DenseBlock(nn.Module):
    """Fully Connected Block: Linear + Activation + BatchNorm"""
    def __init__(self, in_features, out_features, activation='relu'):
        super().__init__()
        
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class CustomModel(nn.Module):
    def __init__(self, input_shape=2500, in_channels=2):
        super().__init__()

        # Branch 1 (input_1)
        self.s1_c1 = ConvBlock(in_channels=in_channels, out_channels=128, kernel_size=9)
        self.s1_c2 = ConvBlock(in_channels=128, out_channels=128, kernel_size=9)
        self.s1_c3 = ConvBlock(in_channels=128, out_channels=128, kernel_size=9)
        self.s1_c4 = ConvBlock(in_channels=128, out_channels=256, kernel_size=5)
        self.s1_c5 = ConvBlock(in_channels=256, out_channels=256, kernel_size=5)
        self.s1_c6 = ConvBlock(in_channels=256, out_channels=256, kernel_size=5)
        self.s1_c7 = ConvBlock(in_channels=256, out_channels=256, kernel_size=5)
        self.s1_c8 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3)
        self.s1_c9 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3)
        
        s1_input = input_shape
        for _ in range(8):
            s1_input = (s1_input // 2) + 1
        
        self.s1_fc1 = DenseBlock(256*s1_input, 256)
        self.s1_fc2 = DenseBlock(256, 256)
        self.s1_fc3 = DenseBlock(256, 128)
        self.s1_output = nn.Linear(128, 128)  # Softmax layer

        # Branch 2 (input_2)
        self.s2_c1 = ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=9)
        self.s2_c2 = ConvBlock(in_channels=32, out_channels=32, kernel_size=9)
        self.s2_c3 = ConvBlock(in_channels=32, out_channels=32, kernel_size=9)
        self.s2_c4 = ConvBlock(in_channels=32, out_channels=512, kernel_size=5)
        self.s2_c5 = ConvBlock(in_channels=512, out_channels=512, kernel_size=5)
        self.s2_c6 = ConvBlock(in_channels=512, out_channels=512, kernel_size=5)
        self.s2_c7 = ConvBlock(in_channels=512, out_channels=512, kernel_size=5)
        self.s2_c8 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3)
        self.s2_c9 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3)

        self.s2_fc1 = DenseBlock(512*s1_input , 256)
        self.s2_fc2 = DenseBlock(256, 256)
        self.s2_fc3 = DenseBlock(256, 128)
        self.s2_output = nn.Linear(128, 128)  # Softmax layer

        # Merge & Final Layers
        self.merge_fc = DenseBlock(256, 256)  # (75 + 75 from both branches)
        self.merge_fc2 = DenseBlock(256, 128)
        self.final_output = nn.Linear(128, 128)  # Final classification

    def forward(self, x):
        # Pass input_1 through branch 1
        
        x1 = torch.tanh(x)
        
        x1 = self.s1_c1(x1)
        x1 = self.s1_c2(x1)
        x1 = self.s1_c3(x1)
        x1 = self.s1_c4(x1)
        x1 = self.s1_c5(x1)
        x1 = self.s1_c6(x1)
        x1 = self.s1_c7(x1)
        x1 = self.s1_c8(x1)
        x1 = self.s1_c9(x1)
        
        x1 = torch.flatten(x1, start_dim=1)
        
        x1 = self.s1_fc1(x1)
        x1 = self.s1_fc2(x1)
        x1 = self.s1_fc3(x1)
        x1 = self.s1_output(x1)
        
        x2 = x * (torch.hypot(x[:, :, 0], x[:, :, 1]) > 0.04).unsqueeze(-1)
        x2 = (x2 - (-9.26532e-05)) / 0.060868427
        
        # Pass input_2 through branch 2
        x2 = self.s2_c1(x2)
        x2 = self.s2_c2(x2)
        x2 = self.s2_c3(x2)
        x2 = self.s2_c4(x2)
        x2 = self.s2_c5(x2)
        x2 = self.s2_c6(x2)
        x2 = self.s2_c7(x2)
        x2 = self.s2_c8(x2)
        x2 = self.s2_c9(x2)
        x2 = torch.flatten(x2, start_dim=1)
        x2 = self.s2_fc1(x2)
        x2 = self.s2_fc2(x2)
        x2 = self.s2_fc3(x2)
        x2 = self.s2_output(x2)

        # Merge both branches
        merged = torch.cat((x1, x2), dim=1)  # Concatenate outputs
        merged = self.merge_fc(merged)
        merged = self.merge_fc2(merged)
        final_output = self.final_output(merged)

        return final_output


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

        input_shape: Iterable[int] = [9, 1024]
        self.normalize_embeddings: bool = False
        self.classifier = Classifier(128, n_classes)
        self.n_classes = n_classes
        

        self.input_shape = input_shape
        c_in, t_in = self.input_shape

        
        self.embedder = CustomModel(input_shape=seq_len//2, in_channels=9)
        
        
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

    def metric_step(
        self, embeddings: torch.Tensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        if self.w_metric_loss <= 0.0:
            return 0.0


        return 0.0

    def class_step(
        self, embeddings: torch.Tensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        # Since we have class-disjoint datasets, only compute class loss
        # on the training set.  We know we're working with the train set
        # if `self.embedder.training` is True.
        # if (
        #     self.classifier is None
        #     or self.w_class_loss <= 0.0
        #     or not self.training
        # ):
        #     return 0.0
        
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