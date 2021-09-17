

from typing import Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from loaders import FondosLoader
from config import Dataset
# from loaders import #not implemented
from torchvision import transforms
from PIL import Image
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD

class DataModule(LightningDataModule):
    """
     A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    """
    
    def __init__(self, 
                 data_dir:str,
                 batch_size:int,
                 num_workers:int,
                 pin_memory:bool,
                 dataset:Dataset,
                 train_val_test_split_percentage:Tuple[float,float,float]=(0.7,0.3,0.0),
                 input_size=None
                 
                 
                 ):
        
        super().__init__()
        self.data_dir=data_dir
        
        self.train_val_test_split_percentage = train_val_test_split_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_enum=dataset
        self.input_size=input_size
        self.get_dataset()
        

    def get_dataset(self):
        
        
        if self.dataset_enum == Dataset.elementos_presentes:
            NotImplementedError
            # self.dataset=
            # self.in_chans=3
            if self.input_size is None:
                self.input_size=None
            
        elif self.dataset_enum == Dataset.fondos:
            self.dataset=FondosLoader
            self.in_chans=3
            if self.input_size is None:
                self.input_size=600
                
    
        self.transform=transforms.Compose([
                                    transforms.Resize((self.input_size,self.input_size), Image.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[IMAGENET_DEFAULT_MEAN[0],IMAGENET_DEFAULT_MEAN[1],IMAGENET_DEFAULT_MEAN[2]],
                                                         std=[IMAGENET_DEFAULT_STD[0],IMAGENET_DEFAULT_STD[1],IMAGENET_DEFAULT_STD[2]])]
            )
    def prepare_data(self):
     
        
        pass
    
    def setup(self,stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        fulldataset = self.dataset(root=self.data_dir,
                                   transform=self.transform)
        train_val_test_split= [round(split*len(fulldataset)) for split in self.train_val_test_split_percentage]
        if not sum(train_val_test_split)==len(fulldataset):
            train_val_test_split[0]+=1
        self.data_train, self.data_val, self.data_test = random_split(
            fulldataset, train_val_test_split
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
