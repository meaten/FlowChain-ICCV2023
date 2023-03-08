from yacs.config import CfgNode
from typing import Callable


def Build_Metrics(cfg: CfgNode) -> Callable:
    if cfg.DATA.TASK == "TP":
        from metrics.TP_metrics import TP_metrics
        return TP_metrics(cfg)

    if cfg.DATA.TASK == "VP":
        from metrics.VP_metrics import VP_metrics
        return VP_metrics

    if cfg.DATA.TASK == "MP":
        from metrics.MP_metrics import MP_metrics
        return MP_metrics