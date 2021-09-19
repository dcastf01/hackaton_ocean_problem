import torch
import torch.nn as nn

import timm
from timm.models.layers.classifier import create_classifier
class FacebookModels(nn.Module):
    def __init__(self,num_class,name_model):
        super().__init__()
        self.backbone=torch.hub.load("facebookresearch/dino:main" ,name_model)
        num_features=self.backbone.num_features
        _,self.classifier=create_classifier(num_features,num_class,pool_type="avg")
        
    def forward(self,x):
        x=self.backbone(x)
        # x=self.global_pool(x)
        x=self.classifier(x)
        return x