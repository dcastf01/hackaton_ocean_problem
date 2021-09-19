
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from config import  ModelsAvailable
from lit_system import LitSystem
from metrics import get_metrics_collections_base
from timm.models.layers.classifier import create_classifier
from image_classification.models import get_model
from factory_model import FacebookModels
class LitClassifier(LitSystem):
    
    def __init__(self,
                 lr,
                 optim: str,
                 model_name:str,
                 in_chans:int,
                 num_class:int,
                 num_fold:int,
                 num_repeat:int
                 ):
        
        
        super().__init__(lr, optim=optim,num_classes=num_class)
        extras=dict(in_chans=in_chans)
        self.generate_model(model_name,in_chans,num_class)
        # self.model=timm.create_model(model_name,pretrained=True,num_classes=10,**extras)
        self.criterion=torch.nn.CrossEntropyLoss()
        
        self.num_fold=num_fold
        self.num_repeat=num_repeat
        
    def forward(self,x):
        return self.model(x)
    
    def _step(self,batch,metric,prefix=""):
        x,labels=batch
        preds=self.forward(x)
        loss=self.criterion(preds,labels)
        preds=preds.softmax(dim=1)
        
        try:
            metric_value=metric(preds,labels)
            data_dict={prefix+"loss":loss,**metric_value}
            self.insert_each_metric_value_into_dict(data_dict,prefix="")
        except Exception as e:
            print(e)
            
        return loss
    
    def training_step(self, batch,batch_idx):
        loss=self._step(batch,self.train_metrics_base)
      
        
        return loss
    
    def validation_step(self, batch,batch_idx):
        loss=self._step(batch,self.valid_metrics_base,"val_")
    
    def test_step(self, batch,batch_idx):
        loss=self._step(batch,self.test_metrics_base,"test_")
     
    
    def generate_model(self,model_name:str,in_chans:int,num_class:int):
        
        if isinstance(model_name,str):
            model_enum=ModelsAvailable[model_name.lower()]
        # print(timm.list_models())
        # print(timm.list_models(pretrained=True))
        
        if model_enum.value in timm.list_models(pretrained=True) and isinstance(num_class,int)  :
            extras=dict(in_chans=in_chans)
            self.model=timm.create_model(
                                        model_enum.value,
                                        pretrained=True,
                                        num_classes=num_class,
                                        **extras
                                        )
        elif model_enum.value in timm.list_models(pretrained=True) and isinstance(num_class,list):
            extras=dict(in_chans=in_chans)
            self.model=timm.create_model(
                                        model_enum.value,
                                        pretrained=True,
                                        num_classes=0,
                                        **extras
                                        )
        elif model_enum==ModelsAvailable.xcits:
            self.model=get_model( "XciT","S",
                                 pretrained="checkpoints/xcit_small_24_p16_224_dist.pth",
                                 image_size=448)
        elif model_enum.name[0:4]==ModelsAvailable.dino_xcit_medium_24_p16.name[0:4]:
            self.model=FacebookModels(num_class=num_class,name_model=model_enum.value)
class LitClassifierTwoInOne(LitClassifier):
    
    def __init__(self, 
                 lr, 
                 optim: str,
                 model_name: str, 
                 in_chans: int, 
                 num_class: list,
                 num_fold: int, 
                 num_repeat: int):
        
        super().__init__(lr, optim, model_name, in_chans, num_class, num_fold, num_repeat)
        self.train_metrics_base0=get_metrics_collections_base(prefix="train0",num_classes=num_class[0])
        self.valid_metrics_base0=get_metrics_collections_base(prefix="valid0",num_classes=num_class[0])
        self.train_metrics_base1=get_metrics_collections_base(prefix="train1",num_classes=num_class[1])
        self.valid_metrics_base1=get_metrics_collections_base(prefix="valid1",num_classes=num_class[1])
        self.modified_model(num_class)
        weights_0 = torch.ones(5)
        weights_1 = torch.ones(6)
        ignore_classes = torch.LongTensor([0])
        weights_0[ignore_classes] = 0.5
        weights_1[ignore_classes] = 0.5
        self.criterion_0=torch.nn.CrossEntropyLoss(weight=weights_0)
        self.criterion_1=torch.nn.CrossEntropyLoss(weight=weights_1)
        
    def modified_model(self,num_class):
        
        _, self.classifier_0 = create_classifier(
            self.model.num_features, num_class[0]+1, pool_type="avg")
        
        _, self.classifier_1 = create_classifier(
            self.model.num_features, num_class[1]+1, pool_type="avg")
        
        print(self.model)
        #añadir otra cabeza másy ver los números de clase por cabeza
        
    def forward(self,x):
        x=self.model(x)  
        result_0=self.classifier_0(x)
        result_1=self.classifier_1(x)
        return   result_0,result_1
    def _step(self,batch,metric0,metric1,prefix=""):
        
        x,labels,dataset_id=batch
        preds_0,preds_1=self.forward(x)
        #lo siguiente será realizar un clasificador binario
        tensor_template_0=torch.zeros(labels.shape[0]).to(self.device)
        tensor_label_useless_0=torch.full(labels.shape,0).to(self.device)
        labels0=torch.where(dataset_id==tensor_template_0,labels,tensor_label_useless_0)
        tensor_label_useless_1=torch.full(labels.shape,0).to(self.device)
        tensor_template_1=torch.ones(labels.shape[0]).to(self.device)
        labels_1=torch.where(dataset_id==tensor_template_1,labels,tensor_label_useless_1)
       
        loss_0=self.criterion_0(preds_0,labels0)
        preds_0=preds_0.softmax(dim=1)
        
        loss_1=self.criterion_1(preds_1,labels_1)
        preds_1=preds_1.softmax(dim=1)
        
        loss=loss_0*0.5+loss_1*0.5
        try:
            metric_value0=metric0(preds_0,labels0)
            metric_value1=metric1(preds_1,labels_1)
            
            data_dict={
                prefix+"loss":loss,
                prefix+"loss_0":loss_0,
                prefix+"loss_1":loss_1,
                **metric_value0,**metric_value1}
            self.insert_each_metric_value_into_dict(data_dict,prefix="")
        except Exception as e:
            print(e)
            
        return loss
    
    def training_step(self, batch,batch_idx):
        loss=self._step(batch,self.train_metrics_base0,self.train_metrics_base1)
      
        
        return loss
    
    def validation_step(self, batch,batch_idx):
        loss=self._step(batch,self.valid_metrics_base0,self.valid_metrics_base1,"val_")
    
    # def test_step(self, batch,batch_idx):
    #     loss=self._step(batch,self.test_metrics_base,"test_")