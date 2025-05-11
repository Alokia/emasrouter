from typing import List, Dict, Tuple, Optional
from utils.graph_utils import get_isolated_nodes
import torch
import torch.nn.functional as F

configs = {
    "_empty_nodes_penalty": True,
    "_isolated_nodes_penalty": True,
}


class LossFunction:
    def __init__(self, lambda_cost: float = 5.0, logger=None):
        self.lambda_cost = lambda_cost
        self.logger = logger

    @staticmethod
    def _isolated_nodes_penalty(selected_roles: List[Dict], selected_edges: List[Tuple], **kwargs):
        if len(selected_roles) == 1:
            return True, None
        isolated_nodes = get_isolated_nodes(selected_roles, selected_edges)
        if len(isolated_nodes) != 0:
            return False, -len(isolated_nodes)
        return True, None

    @staticmethod
    def _empty_nodes_penalty(selected_roles: List[Dict], **kwargs):
        if len(selected_roles) == 0:
            return False, -10.0
        return True, None

    def pre_calculate(self, log_probs, selected_roles, selected_edges, selected_llms) -> Tuple[bool, Optional[float]]:
        for k, v in configs.items():
            if v:
                func = getattr(self, k)
                is_valid, penalty = func(selected_roles=selected_roles,
                                         selected_edges=selected_edges,
                                         selected_llms=selected_llms)
                if not is_valid:
                    return False, -log_probs * penalty
        return True, None

    def calculate(self, log_probs, total_cost, is_solved, num_selected_roles, **kwargs):
        reward = is_solved - self.lambda_cost * total_cost - 0.05 * num_selected_roles
        self.logger.info(
            f":: Reward: {reward}\n"
        )
        return -log_probs * reward

    def calculate_critic(self, log_probs, total_cost, is_solved, state_value, num_selected_roles, **kwargs):
        reward = is_solved - self.lambda_cost * total_cost - 0.05 * num_selected_roles
        advantage = (reward - state_value).detach()
        loss_actor = - advantage * log_probs
        loss_critic = F.mse_loss(state_value, torch.tensor([reward], device=state_value.device))
        loss = loss_actor + loss_critic
        self.logger.info(
            f"\n:: Reward: {reward}" +
            f"\n:: State Value: {state_value}" +
            f"\n:: Advantage: {advantage}" +
            f"\n:: Actor: {loss_actor}" +
            f"\n:: Critic: {loss_critic}"
        )
        return loss

    def calculate_task_loss(self, task_scores, task_labels):
        task_loss = F.cross_entropy(task_scores, task_labels)
        self.logger.info(
            f"\n:: Task Loss: {task_loss}"
        )
        return task_loss

    def __call__(self, critic: bool = True, **kwargs):
        if not critic:
            return self.calculate(**kwargs)
        else:
            return self.calculate_critic(**kwargs)

