from .base_role import BaseRole
import yaml
import os

__all__ = ["role_pool"]


def get_role_pool(role_path: str = None):
    if role_path is None:
        root_path = os.path.abspath(__file__)
        role_path = os.path.join(os.path.dirname(root_path), "roles.yaml")
    with open(role_path, "r") as f:
        role_cfg = yaml.safe_load(f)

    ins_role_pool = {}
    for domain, roles in role_cfg.items():
        ins_role_pool[domain] = []
        for role in roles:
            ins_role_pool[domain].append(
                {
                    "name": role["role"],
                    "description": role["description"],
                    "instance": BaseRole(**role),
                }
            )

    return ins_role_pool


role_pool = get_role_pool()
