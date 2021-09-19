
from lit_classifier import LitClassifier
from config import CONFIG
import os
def load_system(checkpoint_name,root_path=CONFIG.PATH_CHECKPOINT):
    path_ckpt=os.path.join(root_path,checkpoint_name)
    system = LitClassifier.load_from_checkpoint(path_ckpt)
    
    return system