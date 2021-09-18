
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
        target=target+1
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
        target=target+1
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target,0
