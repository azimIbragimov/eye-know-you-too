import torch

from abc import abstractmethod

class BaseDataset(torch.nn.Module):
    @abstractmethod
    def prepare_data(self):
        return NotImplemented
    
    @abstractmethod
    def setup(self, stage):
        return NotImplemented
    
    @abstractmethod
    def train_dataloader(self):
        return NotImplemented
    
    @abstractmethod
    def val_dataloader(self):
        return NotImplemented

    @abstractmethod
    def test_dataloader(self):
        return NotImplemented
    
    def download_and_process_gazebase(self):
        return NotImplemented

    