from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Optional
from class_registry import ClassRegistry
from class_registry.base import AutoRegister

prompt_registry = ClassRegistry("domain")


class BasePrompt(AutoRegister(prompt_registry), ABC):
    def __init__(self):
        pass

    @staticmethod
    def get_role_description(role) -> str:
        return role.description

    @staticmethod
    def get_strategy_prompt(strategy) -> str:
        if strategy is None:
            return ""
        return strategy.strategy_prompt

    @abstractmethod
    def get_task_constraint(self, *args, **kwargs) -> str:
        pass

    def create_query_messages(self, role, strategy, query: str, predecessors_messages: List[str]) -> List[Dict[str, str]]:
        role_description = self.get_role_description(role)
        # task_constraint = self.get_task_constraint()
        task_constraint = ""
        strategy_prompt = self.get_strategy_prompt(strategy)

        messages = [{
            "role": "system",
            "content": role_description + "\n" + task_constraint + "\n" + strategy_prompt
        }, {
            "role": "user",
            "content": "This is the task: " + query
        }]
        if predecessors_messages is not None and len(predecessors_messages) > 0:
            messages.append({
                "role": "user",
                "content": "Following is the output of the other roles: " + "\n".join(predecessors_messages)
            })
        messages.append({
            "role": "user",
            "content": "Please provide your answer:"
        })

        return messages
