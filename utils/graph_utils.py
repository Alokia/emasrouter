from typing import List, Union, Tuple, Dict, Any, Hashable
import itertools


def find_node_with_edge(edges: List[Tuple]) -> List[tuple]:
    # find all nodes in the edges
    nodes = set()
    for edge in edges:
        u, v = edge[0], edge[1]
        nodes.add(u)
        nodes.add(v)
    return list(nodes)


def get_complete_graph_edges(
        nodes: Union[List, int], include_self_loops: bool = False
) -> List[Tuple[int, int]]:
    if isinstance(nodes, int):
        nodes = list(range(nodes))
    edges = list(itertools.product(nodes, repeat=2))

    if not include_self_loops:
        edges = [(u, v) for u, v in edges if u != v]
    return edges


def create_adj_table(
        edges: List[Tuple[Hashable]], drop_self_loops: bool = True, with_strategy: bool = False
) -> Dict[Hashable, List[Hashable]]:
    # 根据一组元组创建邻接表
    graph = {}
    for edge in edges:
        u, v = edge[0], edge[1]
        if drop_self_loops and u == v:
            continue
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        if with_strategy:
            strategy = edge[2]
            graph[u].append((v, strategy))
        else:
            graph[u].append(v)
    return graph


def create_pre_suc_table(
        edges: List[Tuple], drop_self_loops: bool = True
) -> Tuple[Dict[Any, List[Any]], Dict[Any, List[Any]]]:
    # 根据一组元组创建前驱后继表
    predecessors, successors = {}, {}
    for edge in edges:
        u, v = edge[0], edge[1]
        if u == v and drop_self_loops:
            continue
        if v not in predecessors:
            predecessors[v] = []
        if u not in successors:
            successors[u] = []
        # v 的前驱是 u
        predecessors[v].append(u)
        # u 的后继是 v
        successors[u].append(v)
    return predecessors, successors


def find_io_nodes(
        nodes: List[Any], edges: List[Tuple], drop_self_loops: bool = True
) -> Tuple[List, List, List, List]:
    # 找到输入输出节点
    predecessors, successors = create_pre_suc_table(edges, drop_self_loops=drop_self_loops)
    input_nodes, output_nodes, middle_nodes, isolated_nodes = [], [], [], []
    for node in nodes:
        # 没有前驱和后继的节点是孤立节点
        if node not in predecessors and node not in successors:
            isolated_nodes.append(node)
        # 有后继，没有前驱的节点是输入节点
        elif node in successors and node not in predecessors:
            input_nodes.append(node)
        # 有前驱，没有后继的节点是输出节点
        elif node in predecessors and node not in successors:
            output_nodes.append(node)
        # 有前驱和后继的节点是中间节点
        else:
            middle_nodes.append(node)

    return input_nodes, output_nodes, middle_nodes, isolated_nodes


def has_cycle_directed(
        edges: List[Tuple], drop_self_loops: bool = True
) -> bool:
    # 判断有向图是否有环
    graph = create_adj_table(edges, drop_self_loops=drop_self_loops)

    visited = set()
    recursion_stack = set()  # to check if a node is in the current recursion stack

    def dfs(node):
        visited.add(node)
        recursion_stack.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in recursion_stack:
                # if the neighbor is in the recursion stack, there is a cycle
                return True

        recursion_stack.remove(node)
        return False

    # check all nodes
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False


def get_isolated_nodes(nodes: List, edges: List[Tuple],
                       is_dict_node: bool = True, drop_self_loop: bool = True) -> List:
    # 获取孤立节点
    count = {}
    res = []
    for edge in edges:
        u, v = (edge[0], edge[1]) if not is_dict_node else (edge[0]["name"], edge[1]["name"])
        if drop_self_loop and u == v:
            continue
        count[u] = count.get(u, 0) + 1
        count[v] = count.get(v, 0) + 1
    for node in nodes:
        n = node if not is_dict_node else node["name"]
        if n not in count:
            res.append(node)
    return res


def make_elements(
        edges: List[Tuple], nodes: List = None, input_nodes: List = None,
        output_nodes: List = None, middle_nodes: List = None,
        isolated_nodes: List = None, use_class: bool = True,
        node_label: Dict = None
) -> List[Dict[str, Any]]:
    # 构建 dash 元素列表，用于图的可视化
    if nodes is None and input_nodes is None and output_nodes is None and middle_nodes is None and isolated_nodes is None:
        raise ValueError(
            "At least one of nodes, input_nodes, output_nodes, middle_nodes, or isolated_nodes must be provided."
        )
    if nodes is not None and input_nodes is None and output_nodes is None and middle_nodes is None and isolated_nodes is None:
        raise ValueError(
            "If nodes is provided, input_nodes, output_nodes, middle_nodes, and isolated_nodes must be None."
        )
    if nodes is not None and use_class:
        input_nodes, output_nodes, middle_nodes, isolated_nodes = find_io_nodes(
            nodes, edges, drop_self_loops=True
        )
        nodes_dict = {
            "input": input_nodes,
            "output": output_nodes,
            "midpoint": middle_nodes,
            "single": isolated_nodes
        }
    elif nodes is not None and not use_class:
        nodes_dict = {"single", nodes}
    else:
        nodes_dict = {
            "input": input_nodes if input_nodes is not None else [],
            "output": output_nodes if output_nodes is not None else [],
            "midpoint": middle_nodes if middle_nodes is not None else [],
            "single": isolated_nodes if isolated_nodes is not None else []
        }

    elements = []
    for _cls, _node_list in nodes_dict.items():
        for _node in _node_list:
            if node_label is not None and _node in node_label:
                label = str(node_label[_node])
            else:
                label = str(_node)
            ele = {'data': {'id': str(_node), 'label': label}, 'classes': _cls}
            elements.append(ele)

    for source, target, strategy in edges:
        elements.append({
            'data': {
                'id': f"{str(source)}-{str(target)}",
                'source': str(source),
                'target': str(target),
                'label': str(strategy)
            }
        })
    return elements


def make_mermaid(nodes: List, edges: List[Tuple]):
    adj_table = create_adj_table(edges, drop_self_loops=False, with_strategy=True)
    lines = ["flowchart TD"]

    connected_nodes = set()
    for source, targets in adj_table.items():
        for target, strategy in targets:
            source_id = source.role.replace(" ", "_")
            target_id = target.role.replace(" ", "_")
            strategy_id = strategy.strategy.replace(" ", "_")
            lines.append(f"    {source_id}--->|{strategy_id}|{target_id}")
            connected_nodes.add(source_id)
            connected_nodes.add(target_id)

    for node in nodes:
        node_id = node.role.replace(" ", "_")
        if node_id not in connected_nodes:
            lines.append(f"    {node_id}")  # 没有边，单独列出

    return "\n".join(lines)
