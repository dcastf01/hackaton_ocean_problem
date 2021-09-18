import torch
import torchmetrics
from torchmetrics import MetricCollection,Recall,Accuracy

def get_metrics_collections_base(prefix,num_classes
                            # device="cuda" if torch.cuda.is_available() else "cpu",
                            
                            ):
    
    metrics = MetricCollection(
    {
        "Accuracy":Accuracy(),
        "Top_3":Accuracy(top_k=3),
        
    },
    prefix=prefix
             )
    if num_classes is not None:
        
        MetricCollection({"Recall": Recall(num_classes)})
        
    return metrics