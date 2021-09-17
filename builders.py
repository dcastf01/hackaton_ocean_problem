import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from lit_classifier import LitClassifier
from callbacks import  SplitDatasetWithKFoldStrategy
from datamodule import DataModule
from config import CONFIG, Dataset,TargetModel


def build_dataset(root_path:str,dataset_name:str=CONFIG.dataset_name,
                  batch_size:int=CONFIG.batch_size):
    
    
    dataset_enum=Dataset[dataset_name]

    data_module=DataModule(data_dir=os.path.join(root_path,dataset_enum.value),
                                            batch_size=batch_size,
                                            dataset=dataset_enum,
                                            num_workers=CONFIG.NUM_WORKERS,
                                            pin_memory=True)
    data_module.setup()
    return data_module

def get_callbacks(config:CONFIG,dm,only_train_and_test=False):
    #callbacks
    
    early_stopping=EarlyStopping(monitor='_val_loss',
                                 mode="min",
                                patience=10,
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
            # grad_cam,
            learning_rate_monitor,
            # early_stopping,
            split_dataset,
                ]
    
    else:
        callbacks=[
            # grad_cam,
           
            learning_rate_monitor,
            # early_stopping,
            
                ]
  
    return callbacks

def get_system(config,dm,num_fold=0,num_repeat=0):
    dataset_name=config.dataset_name
    dataset_enum=Dataset[dataset_name]
    if config.target_name==TargetModel.classifier_model.name:
        
        system=LitClassifier(
            model_name=config.experiment_name,
            lr=config.lr,
            optim=config.optim_name,
            in_chans=dm.in_chans,
            num_fold=num_fold,
            num_repeat=num_repeat
                             )
   
        
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
