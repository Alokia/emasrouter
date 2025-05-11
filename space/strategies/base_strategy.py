from abc import ABC, abstractmethod
from class_registry import ClassRegistry
from class_registry.base import AutoRegister
from typing import List, Tuple, Dict, Union

strategy_registry = ClassRegistry("strategy")


class BaseStrategy(AutoRegister(strategy_registry), ABC):
    strategy: str = None
    description: str = None
    self_loop: bool = False
    level: int = 5

    def __init__(self, src=None, tgt=None, **kwargs):
        self.src = src
        self.tgt = tgt

    def set_linked_nodes(self, src, tgt):
        self.src = src
        self.tgt = tgt

    @property
    def linked_nodes(self):
        return [self.src, self.tgt]

    def __str__(self):
        return self.strategy

    def __repr__(self):
        return f"{self.__class__.__name__}({self.strategy})"

    @abstractmethod
    def strategy_prompt(self, *args, **kwargs) -> str:
        """Create a system prompt for the strategy.

        Returns:
            Dict. The system prompt for the strategy.
                {"user": "system", "content": "......"}
        """
        pass

    @abstractmethod
    def execute(
            self, query: str, nodes: List = None, *args, **kwargs
    ) -> Tuple[List[Union[Tuple[str, str], Dict[str, str]]], bool]:
        """Execute the strategy.

        Parameters:
            nodes: List[BaseNode]. This parameter contains two nodes. The directed edge points from the first node to the second node,
                and information is exchanged between the two nodes using this strategy.
            query: str. The query to be executed.

        Returns:
            Tuple[List[Tuple[str, str]], bool].
            The first element of the tuple is a list of messages exchanged between the two nodes.
                For example, [(node1_message1, node2_message1), (node1_message2, node2_message2), ...]
            The second element of the tuple is a boolean value indicating whether the strategy was executed.
                If the strategy was executed successfully, the value is True; otherwise, the value is False.
                The edge between the two nodes will be deleted if the value is True.
        """
        pass
