from .base_strategy import strategy_registry
from .edge.ChainStrategy import ChainStrategy
from .edge.DebateStrategy import DebateStrategy
from .loop.CoTStrategy import CoTStrategy
from .loop.ReflectionStrategy import ReflectionStrategy
import yaml
import os

__all__ = ["edge_strategy_pool", "loop_strategy_pool"]


def get_strategy_pool(strategy_pool_path: str = None):
    if strategy_pool_path is None:
        root_path = os.path.abspath(__file__)
        strategy_pool_path = os.path.join(os.path.dirname(root_path), "strategies.yaml")
    with open(strategy_pool_path, "r") as f:
        strategy_pool_cfg = yaml.safe_load(f)

    ins_pool = {}
    for key, strategy_cfg in strategy_pool_cfg.items():
        for cfg in strategy_cfg:
            for name, params in cfg.items():
                temp = {"name": name, "instance": strategy_registry.get(name, **params)}
                temp["description"] = temp["instance"].description
                if key not in ins_pool:
                    ins_pool[key] = []
                ins_pool[key].append(temp)

    return ins_pool["Edge"], ins_pool["Loop"]


edge_strategy_pool, loop_strategy_pool = get_strategy_pool()
