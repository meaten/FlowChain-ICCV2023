from yacs.config import CfgNode
from typing import Dict
import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path
from models.TP.TP_models import Build_TP_model


def Build_Model(cfg: CfgNode) -> nn.Module:
    if cfg.MODEL.TYPE == "GT":
        return GT(cfg)
    elif cfg.MODEL.TYPE == "COPY_LAST":
        return COPY_LAST(cfg)

    if cfg.DATA.TASK == "TP":
        return Build_TP_model(cfg)
    else:
        raise(ValueError)


class ModelTemplate(nn.Module):
    def __init__(self) -> None:
        super(ModelTemplate, self).__init__()
        
    def predict_from_new_obs(self, data_dict: Dict, time_step: int) -> Dict:
        return data_dict

    def predict(self, data_dict, return_prob=False):
        pass

    def update(self, data_dict):
        pass
    
    def load(self, path: Path = None) -> bool:
        pass
    
    def save(self, epoch: int = 0, path: Path = None) -> int:
        pass

class GT(ModelTemplate):
    def __init__(self, cfg: CfgNode) -> None:
        super(GT, self).__init__()

    def predict(self, data_dict, return_prob=False) -> Dict:
        data_dict[("pred", 0)] = deepcopy(data_dict["gt"])
        return data_dict

class COPY_LAST(ModelTemplate):
    def __init__(self, cfg: CfgNode) -> None:
        super(COPY_LAST, self).__init__()
        self.pred_len = cfg.DATA.PREDICT_LENGTH

        self.task = cfg.DATA.TASK

    def predict(self, data_dict, return_prob=False) -> Dict:
        size = data_dict["gt"].size()
        data_dict[("pred", 0)] = data_dict["obs"][:, -1:].expand(size).contiguous()
        return data_dict
    
    


    