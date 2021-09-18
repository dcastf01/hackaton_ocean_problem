import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torchvision import transforms
from torchvision.transforms.autoaugment import _get_transforms
from lit_classifier import LitClassifier,LitClassifierTwoInOne
from callbacks import  SplitDatasetWithKFoldStrategy
from datamodule import DataModule
from config import CONFIG, Dataset,TargetModel,AvailableTransforms
from timm.data.transforms_factory import transforms_imagenet_train,transforms_imagenet_eval

def build_dataset(root_path:str,dataset_name:str=CONFIG.dataset_name,
                  batch_size:int=CONFIG.batch_size,transform_fn_train=None,transform_fn_val=None):
    
    
    dataset_enum=Dataset[dataset_name]
    data_module=DataModule(
        root_path=root_path,
        data_dir=dataset_enum.value,
                           transform_fn_train=transform_fn_train,
                           transform_fn_val=transform_fn_val,
                                            batch_size=batch_size,
                                            dataset=dataset_enum,
                                            num_workers=CONFIG.NUM_WORKERS,
                                            pin_memory=True)
    data_module.setup()
    return data_module

def get_transforms(transforms_name:str):
    transforms_name=AvailableTransforms[transforms_name]
    transformer_string=transforms_name.name
    splitter_transformer=transformer_string.split("_")
    img_size=int(splitter_transformer[0][1:])
    hflip=float(splitter_transformer[1])/100
    vflip=float(splitter_transformer[2])/100
    color_jitter=float(splitter_transformer[3])/100
    auto_augment=str(splitter_transformer[4])
    if auto_augment=="none":
        auto_augment=None
    if transforms_name in AvailableTransforms:
    # [

        # AvailableTransforms.p448_50_50_40_rand,
        #                    AvailableTransforms.p448_50_0_0_rand,
        #                    AvailableTransforms.p448_50_50_40_augmix,
        #                    AvailableTransforms.p600_50_0_40_none
                        #    ]:
        transform_train=transforms_imagenet_train(
            img_size=img_size,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            auto_augment=auto_augment,  
        )
        transform_val=transforms_imagenet_eval(
            img_size=img_size,
        )
    else:
        raise NotImplementedError
    return transform_train,transform_val
def get_callbacks(config:CONFIG,dm,only_train_and_test=False):
    #callbacks
    
    early_stopping=EarlyStopping(monitor='_val_loss',
                                 mode="min",
                                patience=5,
                                 verbose=True,
                                 check_finite =True
                                 )

    checkpoint_callback = ModelCheckpoint(
        monitor='_val_loss',
        dirpath=config.PATH_CHECKPOINT,
        filename= '-{epoch:02d}-{val_loss:.6f}',
        mode="min",
        save_last=True,
        save_top_k=3,
                        )
    learning_rate_monitor=LearningRateMonitor(logging_interval="epoch")
    
    
    # save_result=True if config.num_fold==0 or only_train_and_test  else False
    save_result=True

    if config.num_fold>=1:
        
        split_dataset=SplitDatasetWithKFoldStrategy(folds=config.num_fold,repetitions=config.repetitions,
                                                    dm=dm,
                                                    only_train_and_test=only_train_and_test)

        callbacks=[
            learning_rate_monitor,
            early_stopping,
            split_dataset,
                ]
    
    else:
        callbacks=[
           
            learning_rate_monitor,
            early_stopping,
            
                ]
  
    return callbacks

def get_system(config:CONFIG,dm,num_fold=0,num_repeat=0):
    dataset_name=config.dataset_name
    dataset_enum=Dataset[dataset_name]
    if hasattr(dm.data_train.dataset,"datasets"):
        num_class=[len(dm.data_train.dataset.datasets[0].classes),len(dm.data_train.dataset.datasets[1].classes)]
    else:
        
        num_class=len(dm.data_train.dataset.classes)
    if config.target_name==TargetModel.classifier_model_standar.name:
        
        system=LitClassifier(
            model_name=config.experiment_name,
            lr=config.lr,
            optim=config.optim_name,
            in_chans=dm.in_chans,
            num_class=num_class,
            num_fold=num_fold,
            num_repeat=num_repeat
                             )
    
    elif config.target_name==TargetModel.classifier_model_two_in_one.name and dataset_enum==Dataset.elementos_and_fondos:
        system=LitClassifierTwoInOne(
            model_name=config.experiment_name,
            lr=config.lr,
            optim=config.optim_name,
            in_chans=dm.in_chans,
            num_class=num_class,
            num_fold=num_fold,
            num_repeat=num_repeat
                             )
    else:
        
        raise "algo raro, probablemente seleccionaste los dos dataset"   
    return system

def get_trainer(wandb_logger,callbacks,config):
    
    gpus=[]
    if config.gpu0:
        gpus.append(0)
    if config.gpu1:
        gpus.append(1)
    logging.info( "gpus active",gpus)
    if len(gpus)>=2:
        distributed_backend="ddp"
        accelerator="dpp"
        plugins=DDPPlugin(find_unused_parameters=False)
    else:
        distributed_backend=None
        accelerator=None
        plugins=None

    trainer=pl.Trainer(
                    logger=wandb_logger,
                       gpus=gpus,
                       max_epochs=config.NUM_EPOCHS,
                       precision=config.precision_compute,
                    #    limit_train_batches=0.1, #only to debug
                    #    limit_val_batches=0.1, #only to debug
                    #    limit_test_batches=0.1,
                    #    val_check_interval=1,
                        auto_lr_find=config.AUTO_LR,
                       log_gpu_memory=True,
                    #    distributed_backend=distributed_backend,
                    #    accelerator=accelerator,
                    #    plugins=plugins,
                       callbacks=callbacks,
                       progress_bar_refresh_rate=5,
                       
                       )
    
    return trainer
