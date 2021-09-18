
from torch.utils.data import Dataset
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
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


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