from abc import ABC, abstractmethod
import shortuuid
from typing import List, Optional, Tuple, Any, Dict
from space.llms.base_llm import BaseLM
from space.strategies.base_strategy import BaseStrategy


class _StrategyConnection:
    """refer to https://github.com/The-Pocket/PocketFlow/blob/main/pocketflow/__init__.py
    """

    def __init__(self, src: 'BaseNode', strategy: BaseStrategy):
        self.src = src
        self.strategy = strategy

    def __rshift__(self, tgt: 'BaseNode'):
        if self.src is tgt:
            self.src.loop_strategy = self.strategy
        else:
            self.src.add_successor_with_edge(tgt, self.strategy)
            tgt.add_predecessor_with_edge(self.src, self.strategy)
        self.strategy.set_linked_nodes(self.src, tgt)


class BaseNode(ABC):
    """refer to
    https://github.com/The-Pocket/PocketFlow/blob/main/pocketflow/__init__.py
    https://github.com/metauto-ai/GPTSwarm/tree/main/swarm/graph

    BaseNode 只负责管理节点的前驱和后继关系。
    每个节点都拥有 execute 方法，负责执行节点的操作，执行节点前会调用 pre_process 方法，执行后会调用 post_process 方法。
    其中 pre_process 用于处理节点的输入，post_process 用于处理节点的输出。
    pre_process 应该处理前驱节点的输出，一般用于聚合前驱节点的输出信息。
    post_process 应该处理该节点的输出。
    """

    def __init__(self, node_id: Optional[str] = None, role: str = None, llm: BaseLM = None):
        self.id: str = node_id if node_id is not None else shortuuid.ShortUUID().random(length=4)
        self.llm: BaseLM = llm
        self.role: str = role

        self.loop_strategy: Optional[BaseStrategy] = None

        self.spatial_predecessors: List['BaseNode'] = []
        self.spatial_successors: List['BaseNode'] = []
        self.temporal_predecessors: List['BaseNode'] = []
        self.temporal_successors: List['BaseNode'] = []

        self.spatial_predecessors_with_edge: List[Tuple['BaseNode', BaseStrategy]] = []
        self.spatial_successors_with_edge: List[Tuple['BaseNode', BaseStrategy]] = []
        self.temporal_predecessors_with_edge: List[Tuple['BaseNode', BaseStrategy]] = []
        self.temporal_successors_with_edge: List[Tuple['BaseNode', BaseStrategy]] = []

        self.spatial_predecessor_messages: Dict[str, Dict] = {}
        self.temporal_predecessor_messages: Dict[str, Dict] = {}

        self.inputs: List[Any] = []
        self.outputs: str = ""
        self.raw_inputs: List[Any] = []

        self.last_memory: Dict[str, Any] = {'inputs': [], 'outputs': [], 'raw_inputs': []}

    def add_predecessor(self, node: 'BaseNode', st='spatial'):
        if st == 'spatial' and node not in self.spatial_predecessors:
            self.spatial_predecessors.append(node)
            node.spatial_successors.append(self)
        elif st == 'temporal' and node not in self.temporal_predecessors:
            self.temporal_predecessors.append(node)
            node.temporal_successors.append(self)

    def add_predecessor_with_edge(self, node: 'BaseNode', edge: BaseStrategy, st='spatial'):
        if st == 'spatial' and (node, edge) not in self.spatial_predecessors_with_edge:
            self.spatial_predecessors_with_edge.append((node, edge))
            node.spatial_successors_with_edge.append((self, edge))
        elif st == 'temporal' and (node, edge) not in self.temporal_predecessors_with_edge:
            self.temporal_predecessors_with_edge.append((node, edge))
            node.temporal_successors_with_edge.append((self, edge))
        self.add_predecessor(node, st=st)

    def add_successor(self, node: 'BaseNode', st='spatial'):
        if st == 'spatial' and node not in self.spatial_successors:
            self.spatial_successors.append(node)
            node.spatial_predecessors.append(self)
        elif st == 'temporal' and node not in self.temporal_successors:
            self.temporal_successors.append(node)
            node.temporal_predecessors.append(self)

    def add_successor_with_edge(self, node: 'BaseNode', edge: BaseStrategy, st='spatial'):
        if st == 'spatial' and (node, edge) not in self.spatial_successors_with_edge:
            self.spatial_successors_with_edge.append((node, edge))
            node.spatial_predecessors_with_edge.append((self, edge))
        elif st == 'temporal' and (node, edge) not in self.temporal_successors_with_edge:
            self.temporal_successors_with_edge.append((node, edge))
            node.temporal_predecessors_with_edge.append((self, edge))
        self.add_successor(node, st=st)

    def remove_predecessor(self, node: 'BaseNode', st='spatial'):
        if st == 'spatial' and node in self.spatial_predecessors:
            self.spatial_predecessors.remove(node)
            node.spatial_successors.remove(self)
        elif st == 'temporal' and node in self.temporal_predecessors:
            self.temporal_predecessors.remove(node)
            node.temporal_successors.remove(self)

    def remove_predecessor_with_edge(self, node: 'BaseNode', edge: BaseStrategy, st='spatial'):
        if st == 'spatial' and (node, edge) in self.spatial_predecessors_with_edge:
            self.spatial_predecessors_with_edge.remove((node, edge))
            node.spatial_successors_with_edge.remove((self, edge))
        elif st == 'temporal' and (node, edge) in self.temporal_predecessors_with_edge:
            self.temporal_predecessors_with_edge.remove((node, edge))
            node.temporal_successors_with_edge.remove((self, edge))
        self.remove_predecessor(node, st=st)

    def remove_successor(self, node: 'BaseNode', st='spatial'):
        if st == 'spatial' and node in self.spatial_successors:
            self.spatial_successors.remove(node)
            node.spatial_predecessors.remove(self)
        elif st == 'temporal' and node in self.temporal_successors:
            self.temporal_successors.remove(node)
            node.temporal_predecessors.remove(self)

    def remove_successor_with_edge(self, node: 'BaseNode', edge: BaseStrategy, st='spatial'):
        if st == 'spatial' and (node, edge) in self.spatial_successors_with_edge:
            self.spatial_successors_with_edge.remove((node, edge))
            node.spatial_predecessors_with_edge.remove((self, edge))
        elif st == 'temporal' and (node, edge) in self.temporal_successors_with_edge:
            self.temporal_successors_with_edge.remove((node, edge))
            node.temporal_predecessors_with_edge.remove((self, edge))
        self.remove_successor(node, st=st)

    def clear_connections(self):
        self.spatial_predecessors: List['BaseNode'] = []
        self.spatial_successors: List['BaseNode'] = []
        self.temporal_predecessors: List['BaseNode'] = []
        self.temporal_successors: List['BaseNode'] = []

        self.spatial_predecessors_with_edge: List[Tuple['BaseNode', BaseStrategy]] = []
        self.spatial_successors_with_edge: List[Tuple['BaseNode', BaseStrategy]] = []
        self.temporal_predecessors_with_edge: List[Tuple['BaseNode', BaseStrategy]] = []
        self.temporal_successors_with_edge: List[Tuple['BaseNode', BaseStrategy]] = []

    def update_memory(self):
        self.last_memory['inputs'] = self.inputs
        self.last_memory['outputs'] = self.outputs
        self.last_memory['raw_inputs'] = self.raw_inputs

    @abstractmethod
    def _pre_process(self, query: str, strategy_prompt: str, **kwargs):
        pass

    @abstractmethod
    def _execute(self, pre_result, strategy_prompt: str, **kwargs):
        pass

    def execute(self, query: str, strategy_prompt: str = "", strategy_message: str = "",
                outputs: Dict[str, Dict] = None, **kwargs):
        assert self.llm is not None, f"{self.__class__.__name__} llm is not set."
        response = self._execute(
            query, strategy_prompt=strategy_prompt,
            strategy_message=strategy_message, outputs=outputs, **kwargs
        )
        self.outputs = response
        return response

    @abstractmethod
    def _post_process(self, query: str, pre_result, exec_result, **kwargs):
        pass

    def __rshift__(self, other: 'BaseNode'):
        if other is self:
            raise ValueError("Cannot connect node to itself.")
        self.add_successor(other)

    def __sub__(self, edge: BaseStrategy):
        if isinstance(edge, BaseStrategy):
            return _StrategyConnection(self, edge)
        raise TypeError("Edge must be a `BaseStrategy`")

    def set_llm(self, llm: BaseLM):
        self.llm = llm

    @property
    def name(self):
        return self.id
