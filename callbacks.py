
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from PIL import Image
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_lightning.callbacks.base import Callback
from seaborn.palettes import color_palette
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.offsetbox as osb
from matplotlib import rcParams as rcp

import wandb
import torchvision.transforms.functional as functional

from config import CONFIG, Dataset
from datamodule import DataModule
from lit_classifier import LitClassifier
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.manifold import TSNE
from itertools import chain
import os

class ConfusionMatrix(Callback):
    def __init__(self,dataloader:DataLoader) -> None:
        super().__init__()
        self.dataloader=dataloader
        self.class_names=self.dataloader.dataset.dataset.classes
    def generate_accuracy_and_upload(self,trainer: 'pl.Trainer',pl_module:'pl.LightningModule'):
        all_predictions=[]
        all_ground_truth_ids=[]
        num_correct = 0
        num_samples = 0
        for batch in self.dataloader:
            image,y,=batch
            y=y.to(pl_module.device)
            with torch.no_grad():
                # results=pl_module(image.to(device=pl_module.device))
                predictions = pl_module(image.to(device=pl_module.device))
                predictions  = predictions.argmax(axis=1)
                
            ground_truth_ids=y.cpu().numpy()
            
            prediction_cpu=predictions.cpu().numpy()
            prediction_list=list(prediction_cpu.tolist())
                # num_correct +=torch.sum(predictions == y)
                # num_samples += predictions.size(0)
            all_predictions.append(prediction_list)
            all_ground_truth_ids.append(ground_truth_ids.tolist())
            # all_ground_truth_ids=np.concatenate(all_ground_truth_ids,ground_truth_ids)
            # all_targets.append()
            # all_results.append(results.softmax(dim=1))
            # all_targets.append(target)
        # predictions=predictions.cpu().numpy()
        # class_names= self.class_names
        all_predictions_flatten = [item for sublist in all_predictions for item in sublist]
        all_ground_truth_ids_flatten = [item for sublist in all_ground_truth_ids for item in sublist]
        return all_ground_truth_ids_flatten,all_predictions_flatten,self.class_names

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        ground_truth,predictions,class_names=self.generate_accuracy_and_upload(trainer,pl_module)
        self.create_confusion_matrix(ground_truth,predictions,class_names)
        return super().on_train_end(trainer, pl_module)
    
    def create_confusion_matrix(self,ground_truth,predictions,class_names):
        
       
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=ground_truth, preds=predictions,
                        class_names=class_names)})
        return
    
class SplitDatasetWithKFoldStrategy(Callback):
    
    def __init__(self,folds,repetitions,dm,only_train_and_test=False) -> None:
        super().__init__()
        self.folds=folds
        self.repetitions=repetitions
        self.train_val_dataset_initial=torch.utils.data.ConcatDataset([dm.data_train,dm.data_val])
        self.only_train_and_test=only_train_and_test
        kf = KFold(n_splits=folds)

        self.indices_folds={}
        
        for fold, (train_ids, test_ids) in enumerate(kf.split(self.train_val_dataset_initial)):
            self.indices_folds[fold]={
                "train_ids":train_ids,
                "test_ids":test_ids
            }
        self.current_fold=0   

    def create_fold_dataset(self,num_fold,trainer,pl_module):
        
        train_ids=self.indices_folds[num_fold]["train_ids"]
        test_ids=self.indices_folds[num_fold]["test_ids"]
        trainer.datamodule.data_train=torch.utils.data.Subset(self.train_val_dataset_initial,train_ids)
        trainer.datamodule.data_val=torch.utils.data.Subset(self.train_val_dataset_initial,test_ids)
    
    def create_all_train_dataset(self,trainer):
        
        trainer.datamodule.data_val=trainer.datamodule.data_test
        trainer.datamodule.data_train=self.train_val_dataset_initial

    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.only_train_and_test:
            self.create_all_train_dataset(trainer)
        else:
            self.create_fold_dataset(pl_module.num_fold,trainer,pl_module)
        return super().on_train_start(trainer, pl_module)
    


class PlotLatentSpace(Callback):
    def __init__(self,dataloader:DataLoader) -> None:
        super(PlotLatentSpace,self).__init__()
        self.dataloader=dataloader
        self.path_to_data=""
        self.each_epoch=10
        

    def get_scatter_plot_with_thumbnails(self,trainer,pl_module,embeddings_2d,filenames):
        """Creates a scatter plot with image overlays.
        """
        # initialize empty figure and add subplot
        fig = plt.figure()
        fig.suptitle('Scatter Plot of the Dataset')
        ax = fig.add_subplot(1, 1, 1)
        # shuffle images and find out which images to show
        shown_images_idx = []
        shown_images = np.array([[1., 1.]])
        iterator = [i for i in range(embeddings_2d.shape[0])]
        np.random.shuffle(iterator)
        for i in iterator:
            # only show image if it is sufficiently far away from the others
            dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
            if np.min(dist) < 2e-10:
                continue
            shown_images = np.r_[shown_images, [embeddings_2d[i]]]
            shown_images_idx.append(i)

        # plot image overlays
        for idx in shown_images_idx:
            thumbnail_size = int(rcp['figure.figsize'][0] * 2.)
            path = os.path.join(self.path_to_data, filenames[idx]+".jpg")
            img = Image.open(path)
            img = functional.resize(img, thumbnail_size)
            img = np.array(img)
            img_box = osb.AnnotationBbox(
                osb.OffsetImage(img, cmap=plt.cm.gray_r),
                embeddings_2d[idx],
                pad=0.2,
            )
            ax.add_artist(img_box)

        # set aspect ratio
        ratio = 1. / ax.get_data_ratio()
        ax.set_aspect(ratio, adjustable='box')
        # Save just the portion _inside_ the second axis's boundaries
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('ax2_figure.png', bbox_inches=extent)

        # Pad the saved area by 10% in the x-direction and 20% in the y-direction
        fig.savefig('ax2_figure_expanded.png', bbox_inches=extent.expanded(1.5, 1.5))
        self.upload_image(trainer,fig,"eliminare")
        
    def upload_image(self,trainer,image,text:str):
       
        trainer.logger.experiment.log({
            f"{text}": [
                wandb.Image(image) 
                ],
            })
    def create_embbedings_2d(self,trainer,pl_module):
        embeddings = []
        filenames = []
        labels=[]
        rng=np.random.default_rng()
        classes_selected=np.array(list(range(0,10,1))    )
        # disable gradients for faster calculations
        pl_module.eval()
        with torch.no_grad():
            for i, (x, y_true) in enumerate(self.dataloader):
                y_true=y_true #debido a que el loader te devuelve tres etiquetas
                # move the images to the gpu
                x = x.to(device=pl_module.device)
                # embed the images with the pre-trained backbone
                y = pl_module.model.forward_features(x)
                y=pl_module.model.global_pool(y)
                y = y.squeeze()
                for embbeding,label in zip(y,y_true):
                    if any([(label == class_selected).all() for class_selected in classes_selected]):
                    # if label in class_selected:
                        # store the embeddings and filenames in lists
                        embeddings.append(torch.unsqueeze(embbeding,dim=0))
                        # filenames = filenames + list(fnames)
                        labels.append(label.item())
                # if i*x.shape[0]>250:
                #     break

        # concatenate the embeddings and convert to numpy
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.cpu().numpy()
        tl=TSNE()
        embeddings_2d=tl.fit_transform(embeddings)
        # projection = random_projection.GaussianRandomProjection(n_components=2)
        # embeddings_2d = projection.fit_transform   (embeddings)
        return embeddings_2d ,labels
    def plot_emmbedings(self,trainer,pl_module,embedding,labels):
        fig=plt.figure(figsize=(10,10))
        number_labels=len(set(labels))
        color_pallete=sns.color_palette("tab20",n_colors=number_labels)#[:number_labels]
        sns.scatterplot(embedding[:,0], embedding[:,1], hue=labels,palette=color_pallete)
        plt.title(f"epoch {pl_module.current_epoch} ")
        fig.savefig('ax2_figureonlyxlabel.png')
        self.upload_image(trainer,fig,"Latent_space")
        
    def on_train_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if pl_module.current_epoch % self.each_epoch==0 or pl_module.current_epoch==0:
            embeddings_2d,filenames=self.create_embbedings_2d(trainer,pl_module)
            self.plot_emmbedings(trainer,pl_module,embeddings_2d,filenames)
                                 
        return super().on_epoch_end(trainer, pl_module)    
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        embeddings_2d,filenames=self.create_embbedings_2d(trainer,pl_module)
        self.plot_emmbedings(trainer,pl_module,embeddings_2d,filenames)
   
        return super().on_train_end(trainer, pl_module)