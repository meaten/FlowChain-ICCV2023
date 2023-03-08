from yacs.config import CfgNode
from typing import Dict, List, Type

from visualization.TP_visualizer import TP_Visualizer, Visualizer


def Build_Visualizer(cfg: CfgNode) -> Type[Visualizer]:
    if cfg.DATA.TASK == "TP":
        return TP_Visualizer(cfg)
