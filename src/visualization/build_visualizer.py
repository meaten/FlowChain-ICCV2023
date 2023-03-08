from yacs.config import CfgNode
from typing import Dict, List, Type

from visualization.TP_visualizer import TP_Visualizer, Visualizer
from visualization.VP_visualizer import VP_Visualizer
from visualization.MP_visualizer import MP_Visualizer


def Build_Visualizer(cfg: CfgNode) -> Type[Visualizer]:
    if cfg.DATA.TASK == "TP":
        return TP_Visualizer(cfg)

    if cfg.DATA.TASK == "VP":
        return VP_Visualizer(cfg)

    if cfg.DATA.TASK == "MP":
        return MP_Visualizer(cfg)