from abc import ABC
from typing import List, Dict, Tuple
from graph.base_node import BaseNode
from space.strategies.base_strategy import BaseStrategy
from collections import deque, defaultdict
from utils.graph_utils import find_io_nodes
from utils.print_utils import print_with_color
from graph.final_node import FinalNode


class BaseGraph(ABC):
    def __init__(self, domain: str = None, dataset: str = None):
        self.domain = domain
        self.final_node: FinalNode = FinalNode(dataset=dataset)

        self.input_nodes: List[BaseNode] = []
        self.output_nodes: List[BaseNode] = []
        self.middle_nodes: List[BaseNode] = []
        self.isolated_nodes: List[BaseNode] = []

        self.nodes: List[BaseNode] = []
        self.edges: List[Tuple[BaseNode, BaseNode, BaseStrategy]] = []
        self.edges_without_self_loop: List[Tuple[BaseNode, BaseNode, BaseStrategy]] = []
        self.adj_table: Dict[BaseNode, List[Tuple[BaseNode, BaseStrategy]]] = {}  # 邻接表

    def add_nodes(self, nodes: List[BaseNode]) -> None:
        self.nodes.extend(nodes)
        for node in nodes:
            if node not in self.adj_table:
                self.adj_table[node] = []

    def add_edge(self, src: BaseNode, tgt: BaseNode, strategy: BaseStrategy):
        if src not in self.nodes:
            self.add_nodes([src])
        if tgt not in self.nodes:
            self.add_nodes([tgt])

        src - strategy >> tgt

        if src is not tgt:
            self.adj_table[src].append((tgt, strategy))
            self.edges_without_self_loop.append((src, tgt, strategy))
        self.edges.append((src, tgt, strategy))

    def create_graph(self, nodes: List[Dict], edges: List[Tuple[Dict, Dict, Dict]], llms: List[Dict]):
        # 清空原有的节点和边
        for node in nodes:
            node["instance"].clear_connections()

        self.add_nodes([node["instance"] for node in nodes])
        for node, llm in zip(nodes, llms):
            node["instance"].set_llm(llm["instance"])
        for edge in edges:
            self.add_edge(src=edge[0]["instance"], tgt=edge[1]["instance"], strategy=edge[2]["instance"])
        self.final_node.set_llm(llms)

    def _sort_edges(self):
        for source, targets in self.adj_table.items():
            self.adj_table[source] = sorted(targets, key=lambda x: x[1].level)

    def _find_io_nodes(self):
        self.input_nodes, self.output_nodes, self.middle_nodes, self.isolated_nodes = find_io_nodes(
            self.nodes, self.edges, drop_self_loops=True)

    def run(self, query: str):
        # self._find_io_nodes()
        self._sort_edges()
        # assert len(self.input_nodes) != 0, "No input nodes found."

        # 计算所有节点的入度
        in_degree: Dict[BaseNode, int] = defaultdict(int)
        for src, tgt, strategy in self.edges_without_self_loop:
            in_degree[tgt] += 1

        # 将所有入度为 0 的节点加入队列
        queue = deque()
        for node in self.nodes:
            if in_degree[node] == 0:
                queue.append(node)

        outputs = {}
        while len(queue) > 0:
            src = queue.popleft()
            # 如果节点没有后继，执行节点
            if len(self.adj_table[src]) == 0:
                print_with_color(f"Executing output node `{src.role}`", color="green")
                response = src.execute(query)
                outputs[src.id] = {"role": src.role, "output": response}
            # 如果节点存在后继，执行策略
            else:
                for tgt, strategy in self.adj_table[src]:
                    print_with_color(
                        f"Executing strategy `{strategy.strategy}` from {src.role} to {tgt.role}", color="green"
                    )
                    strategy.execute(query)
                    # 更新入度
                    in_degree[tgt] -= 1
                    if in_degree[tgt] == 0:
                        queue.append(tgt)
        if len(outputs) > 1:
            print_with_color(f"Executing final node `{self.final_node.role}`", color="green")
            # 使用 FinalNode 聚合 outputs 信息
            output = self.final_node.execute(query, outputs=outputs)
        elif len(outputs) == 1:
            # 如果只有一个输出节点，直接返回
            output = list(outputs.values())[0]["output"]
        else:
            output = "No output found."
        return output
