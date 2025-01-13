import torch

from abc import abstractmethod

class BaseModel(torch.nn.Module):
    @abstractmethod
    def forward(self, x):
        return NotImplemented
    
    @abstractmethod
    def metric_step(self, embeddings: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        return NotImplemented
    
    @abstractmethod
    def class_step(self, embeddings: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        return NotImplemented
    