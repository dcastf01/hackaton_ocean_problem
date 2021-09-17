import torch
import torchmetrics
from torchmetrics import MetricCollection,MeanAbsoluteError , MeanSquaredError,SpearmanCorrcoef,PearsonCorrcoef,Accuracy

def get_metrics_collections_base(prefix,is_regressor:bool=True
                            # device="cuda" if torch.cuda.is_available() else "cpu",
                            
                            ):
    if is_regressor:
        metrics = MetricCollection(
                {
                    "MeanAbsoluteError":MeanAbsoluteError(),
                    "MeanSquaredError":MeanSquaredError(),
                    "SpearmanCorrcoef":SpearmanCorrcoef(),
                    "PearsonCorrcoef":PearsonCorrcoef()          
                },
                prefix=prefix
                )
    else:
         metrics = MetricCollection(
            {
                "Accuracy":Accuracy(),
                "Top_3":Accuracy(top_k=3),
            },
            prefix=prefix
            )
    return metrics