import torch
import os
from enum import Enum
from typing import Union

from dataclasses import dataclass,asdict

from torchvision import transforms

ROOT_WORKSPACE: str=""

class ModelsAvailable(Enum):
    resnet50="resnet50"
    densenet121="densenet121"
    vgg16="vgg16"
    alexnet="alexnet"
    googlenet="googlenet"
    tf_efficientnet_b0="tf_efficientnet_b0"
    tf_efficientnet_b4="tf_efficientnet_b4"
    tf_efficientnet_b7="tf_efficientnet_b7"
    
class Dataset (Enum):
    elementos_presentes="ocean_elements"
    fondos="ocean_v2"
  
class TargetModel(Enum):
    regresor_model=1
    classifier_model=2   
    
class Optim(Enum):
    adam=1
    sgd=2
    
class AvailableTransforms(Enum):
    #imgsize_hflip_vflip_colorjitter_autoaugment
    
    p448_50_50_40_rand=1
    
@dataclass
class CONFIG(object):
    
    experiment=ModelsAvailable.tf_efficientnet_b4
    experiment_name:str=experiment.name
    experiment_net:str=experiment.value
    PRETRAINED_MODEL:bool=True
    only_train_head:bool=False #solo se entrena el head
    
    target_model=TargetModel.classifier_model
    target_name:str=target_model.name
    
    transforms_target=AvailableTransforms.p448_50_50_40_rand
    transforms_name:str=transforms_target.name
    #torch config
    batch_size:int = 32
    dataset=Dataset.fondos
    dataset_name:str=dataset.name
    precision_compute:int=16
    optim=Optim.adam
    optim_name:str=optim.name
    lr:float = 0.01 #cambiar segun modelo y benchmark
    AUTO_LR :bool= False



    num_fold:int=0 #if 0 is not kfold train 
    repetitions:int=1
    
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 4
    SEED:int=1
    # IMG_SIZE:int=28
    NUM_EPOCHS :int= 50
    LOAD_MODEL :bool= True
    SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"/model/checkpoint")

    
    ##data
    root_path:str=r"/home/dcast/hackaton_ocean_problem/data"
    
    gpu0:bool=False  
    gpu1:bool=True
    notes:str="final experiments"
    
    version:int=2
    

def create_config_dict(instance:CONFIG):
    return asdict(instance)


