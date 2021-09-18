

from typing import Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split,ConcatDataset,WeightedRandomSampler
from torchvision import datasets
from loaders import FondosLoader,ElementsLoader,FondosLoaderToCombine,ElementsLoaderToCombine
from config import Dataset
# from loaders import #not implemented
from PIL import Image
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from sampler import ImbalancedDatasetSampler
import torch
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
                 input_size=None,
                 transform_fn_train=None,
                 transform_fn_val=None,
                 root_path:str="/home/dcast/hackaton_ocean_problem/data"
                 
                 ):
        
        super().__init__()
        self.root_path=root_path
        self.data_dir=os.path.join(self.root_path,data_dir)
        
        self.train_val_test_split_percentage = train_val_test_split_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_enum=dataset
        self.input_size=input_size
        self.get_dataset()
        
        self.transform_fn_train=transform_fn_train
        self.transform_fn_val=transform_fn_val
        
        self.in_chans=3
    def get_dataset(self):
        
        
        if self.dataset_enum == Dataset.elementos_presentes:
            self.dataset=ElementsLoader
            self.in_chans=3

        elif self.dataset_enum == Dataset.fondos:
            self.dataset=FondosLoader

                
        elif self.dataset_enum==Dataset.elementos_and_fondos:
            self.dataset=[FondosLoaderToCombine,ElementsLoaderToCombine]
            self.datasets_enums=[Dataset.fondos,Dataset.elementos_presentes]
            
    def prepare_data(self):
     
        
        pass
    
    def setup(self,stage=None):
        # def class_imbalance_sampler(labels):
        #     class_count = torch.bincount(labels.squeeze())
        #     class_weighting = 1. / class_count
        #     sample_weights = class_weighting[labels]
        #     sampler = WeightedRandomSampler(sample_weights, len(labels))
        #     return sampler
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if isinstance(self.dataset,list):
            
            alldatasets=[]
            for dataset_enum,dataset in zip(self.datasets_enums,self.dataset):
                root=os.path.join(self.root_path,dataset_enum.value)
                onedataset=dataset(root=root,
                                    transform=self.transform_fn_train)
                alldatasets.append(onedataset)
            self.fulldataset=ConcatDataset(alldatasets)
            
            train_val_test_split= [round(split*len(self.fulldataset)) for split in self.train_val_test_split_percentage]
            if not sum(train_val_test_split)==len(self.fulldataset):
                train_val_test_split[0]+=1
            self.data_train, self.data_val, self.data_test = random_split(
                self.fulldataset, train_val_test_split
            )
            self.sampler=None
            self.data_val.dataset.transform=self.transform_fn_val
        else:
            
            self.fulldataset = self.dataset(root=self.data_dir,
                                    transform=self.transform_fn_train)
            train_val_test_split= [round(split*len(self.fulldataset)) for split in self.train_val_test_split_percentage]
            if not sum(train_val_test_split)==len(self.fulldataset):
                train_val_test_split[0]+=1
            self.data_train, self.data_val, self.data_test = random_split(
                self.fulldataset, train_val_test_split
            )
        
            # class_sample_count = torch.tensor(
            # [(target == t).sum() for t in torch.unique(target, sorted=True)])
            self.sampler=ImbalancedDatasetSampler(self.data_train)

            self.data_val.dataset.transform=self.transform_fn_val

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            sampler=self.sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            # sampler=self.sampler,
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
