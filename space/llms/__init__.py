from .base_llm import llm_registry
from .openai_chat import OpenAIChat
from .groq_chat import GroqChat
from .request_chat import RequestChat
import yaml
import os

__all__ = ["llm_pool"]


def get_llm_pool(llm_pool_path: str = None):
    if llm_pool_path is None:
        root_path = os.path.abspath(__file__)
        llm_pool_path = os.path.join(os.path.dirname(root_path), "llm_pool.yaml")
    with open(llm_pool_path, "r") as f:
        llm_pool_cfg = yaml.safe_load(f)

    ins_llm_pool = [
        {
            "name": cfg["model"],
            "description": cfg["description"],
            "instance": llm_registry.get(cfg["support"], **cfg["llm_param"], model=cfg["model"]),
        }
        for cfg in llm_pool_cfg
    ]
    return ins_llm_pool


llm_pool = get_llm_pool()
