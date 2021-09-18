
from torch.utils.data import Dataset,ConcatDataset
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from torchvision.transforms.transforms import Resize, ToTensor
from torchvision import transforms
from torchvision.datasets import ImageFolder
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Any, Callable, cast, Dict, List, Optional, Tuple,Iterable
import bisect


class Loader(ImageFolder):
   
    def __init__(self, root: str,
                 transform: Optional[Callable] = None,):  
                     
        super().__init__(root, transform=transform, )

class FondosLoader(Loader):
    def __init__(self, root: str, 
                 transform: Optional[Callable] = None, 
    ):
        super().__init__(root, transform=transform, )
        
class ElementsLoader(Loader):
    def __init__(self, root: str, 
                 transform: Optional[Callable] = None, 
    ):
        super().__init__(root, transform=transform, )
        
class ElementsLoaderToCombine(Loader):
    def __init__(self, root: str, 
                 transform: Optional[Callable] = None, 
    ):
        super().__init__(root, transform=transform, )
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target,1
    
class  FondosLoaderToCombine(Loader):
    def __init__(self, root: str, 
                 transform: Optional[Callable] = None, 
    ):
        super().__init__(root, transform=transform, )
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target,0
class ConcatBothDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)
        
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]