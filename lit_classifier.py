
import pytorch_lightning as pl
import timm
import torch

from config import  ModelsAvailable
from lit_system import LitSystem
from timm.models.layers.classifier import create_classifier
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
        self.modified_model(num_class)
        
    def modified_model(self,num_class):
        
        _, self.classifier_1 = create_classifier(
            self.model.num_features, num_class[0], pool_type="avg")
        
        _, self.classifier_2 = create_classifier(
            self.model.num_features, num_class[0], pool_type="avg")
        
        print(self.model)
        #añadir otra cabeza másy ver los números de clase por cabeza
        
    def forward(self,x):
        x=self.model(x)  
        result_1=self.classifier_1(x)
        result_2=self.classifier_2(x)
        return   result_1,result_2
    def _step(self,batch,metric,prefix=""):
        
        x,labels,dataset_id=batch
        preds_1,preds_2=self.forward(x)
        
        loss=self.criterion(preds_1,labels)
        preds=preds_1.softmax(dim=1)
        
        try:
            metric_value=metric(preds_1,labels)
            data_dict={prefix+"loss":loss,**metric_value}
            self.insert_each_metric_value_into_dict(data_dict,prefix="")
        except Exception as e:
            print(e)
            
        return loss
    