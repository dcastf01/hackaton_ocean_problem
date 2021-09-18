import torch
import os
from enum import Enum
from typing import Union

from dataclasses import dataclass,asdict


ROOT_WORKSPACE: str=""
class ModelsAvailable(Enum):
    resnet50="resnet50"
    densenet121="densenet121"
    vgg16="vgg16"
    alexnet="alexnet"
    googlenet="googlenet"
    tf_efficientnet_b0="tf_efficientnet_b0"
    tf_efficientnet_b4="tf_efficientnet_b4"
    tf_efficientnet_b4_ns="tf_efficientnet_b4_ns"
    tf_efficientnet_b7="tf_efficientnet_b7"
    xcitS="checkpoints/xcit_small_24_p16_224_dist.pth"
    
class Dataset (Enum):
    elementos_presentes="ocean_elements"
    fondos="ocean_v2"
    elementos_and_fondos="son ambos dataset"
class TargetModel(Enum):
    classifier_model_two_in_one=1
    classifier_model_standar=2   
    
class Optim(Enum):
    adam=1
    sgd=2
    
class AvailableTransforms(Enum):
    #imgsize_hflip_vflip_colorjitter_autoaugment
    
    p448_50_50_40_rand=1
    p448_50_0_0_rand=2
    p448_50_50_40_augmix=3
    p600_50_0_40_none=4
    p448_50_30_40_rand=5
    p448_50_30_70_rand=6
    
@dataclass
class CONFIG(object):
    
    experiment=ModelsAvailable.tf_efficientnet_b4_ns
    experiment_name:str=experiment.name
    # experiment_net:str=experiment.value
    PRETRAINED_MODEL:bool=True
    only_train_head:bool=False #solo se entrena el head
    
    target_model=TargetModel.classifier_model_standar
    target_name:str=target_model.name
    
    transforms_target=AvailableTransforms.p448_50_30_40_rand
    transforms_name:str=transforms_target.name
    #torch config
    batch_size:int = 49
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
    LOAD_MODEL :bool= False
    SAVE_MODEL :bool= False
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"/model/checkpoint")

    callback_plot_latent_space:bool=False
    callback_matrix_wandb:bool=False
    ##data
    root_path:str=r"/home/dcast/hackaton_ocean_problem/data"
    
    gpu0:bool=False  
    gpu1:bool=True
    notes:str="final experiments"
    
    version:int=3
    #en la versión 3 se ha añadido un transform al dataset de validación ya que se usaba el de entrenamiento
    

def create_config_dict(instance:CONFIG):
    return asdict(instance)


