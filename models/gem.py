from models.embedding import SentenceEncoder
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import List, Dict
from utils.graph_utils import has_cycle_directed

CONST_EPS = 1e-5


# Actor-Critic Graph-based Edge-level Model (ACGEM)
class ACGEM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, device=None,
                 critic: bool = True, critic_hidden_dim: int = None,
                 latent_dim: int = None,
                 task_classifier_pretrained: str = None, task_classifier_freeze: bool = False):
        super().__init__()
        if device is None:
            device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
        self.device = device
        self.critic = critic
        self.text_encoder = SentenceEncoder()

        # 设置 task classifier，用于训练 query 的隐藏表示
        self.task_classifier = TaskClassifier(embed_dim, latent_dim)
        if task_classifier_pretrained is not None:
            self.task_classifier.load_state_dict(torch.load(task_classifier_pretrained, map_location=device))
        if task_classifier_freeze:
            for param in self.task_classifier.parameters():
                param.requires_grad = False

        self.role_latent_encoder = nn.Linear(embed_dim + latent_dim, embed_dim)

        self.role_selector = RoleSelector(embed_dim, hidden_dim)
        self.strategy_selector = StrategySelector(embed_dim, hidden_dim)
        self.llm_selector = LLMSelector(embed_dim, hidden_dim)

        if self.critic:
            c_hidden_dim = critic_hidden_dim if critic_hidden_dim is not None else hidden_dim
            self.critic_model = CriticModel(embed_dim, c_hidden_dim)

    def forward(self, query: str,
                tasks: List[Dict],
                roles: List[Dict],
                edge_strategies: List[Dict], loop_strategies: List[Dict],
                llms: List[Dict]):
        # [1, embed_dim]
        query_embedding = self.text_encoder([query]).to(self.device)
        # [num_tasks, embed_dim]
        task_embedding = self.text_encoder(self._process_data(tasks)).to(self.device)
        # [num_roles, embed_dim]
        role_embedding = self.text_encoder(self._process_data(roles)).to(self.device)
        # [num_edge_strategies, embed_dim]
        edge_strategy_embedding = self.text_encoder(self._process_data(edge_strategies)).to(self.device)
        # [num_loop_strategies, embed_dim]
        loop_strategy_embedding = self.text_encoder(self._process_data(loop_strategies)).to(self.device)
        # [num_llms, embed_dim]
        llm_embedding = self.text_encoder(self._process_data(llms)).to(self.device)

        # 获得 query 的 latent distribution
        query_latent_embedding, task_scores = self.task_classifier(query_embedding, task_embedding)
        role_embedding = self.role_latent_encoder(
            torch.cat([role_embedding, query_latent_embedding.expand(role_embedding.size(0), -1)], dim=-1)
        )

        # 1. 选择角色
        selected_role_index, role_log_probs = self.role_selector(query_embedding, role_embedding)
        # 空列表如何处理
        if not selected_role_index.sum() > 0:
            return task_scores, [], [], [], role_log_probs, None

        # 2. 选择策略：连边
        selected_role_embedding = role_embedding[selected_role_index]
        selected_edge_tuple, strategy_log_probs = self.strategy_selector(
            query_embedding, selected_role_embedding, edge_strategy_embedding, loop_strategy_embedding)

        # 3. 选择 LLM
        selected_edge_index, selected_edge_embedding = self.strategy_selector.get_selected_embedding(
            edge_strategy_embedding, loop_strategy_embedding, selected_edge_tuple)
        selected_llm_index, llm_log_probs, selected_edge_index, selected_edge_embedding = self.llm_selector(
            query_embedding, selected_role_embedding, selected_edge_index, selected_edge_embedding, llm_embedding)
        selected_llm_embedding = llm_embedding[selected_llm_index]

        log_probs = role_log_probs + strategy_log_probs + llm_log_probs

        # 提取选择的 role、strategy、LLM
        selected_roles = self.role_selector.get_selected_role(roles, selected_role_index)
        selected_edges = self.strategy_selector.get_selected_edges(
            selected_roles, edge_strategies, loop_strategies, selected_edge_tuple)
        selected_llms = self.llm_selector.get_selected_llms(llms, selected_llm_index)

        state_value = None
        if self.critic:
            # 计算 critic 的值
            state_value = self.critic_model(
                query_embedding.detach().clone(), selected_role_embedding.detach().clone(), selected_llm_embedding.detach().clone(),
                selected_edge_index.detach().clone(), selected_edge_embedding.detach().clone()
            )
        return task_scores, selected_roles, selected_edges, selected_llms, log_probs, state_value

    @staticmethod
    def _process_data(data):
        processed_data = ["name: " + d["name"] + ". description: " + d["description"] for d in data]
        return processed_data


class TaskClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query_encoder = nn.Linear(input_dim, hidden_dim)
        self.task_encoder = nn.Linear(input_dim, hidden_dim)

    def forward(self, query_embedding, task_embedding):
        query_embedding = self.query_encoder(query_embedding)
        query_embedding = F.normalize(query_embedding, p=2, dim=-1)
        # latent distribution of query
        q_mean, q_std = torch.mean(query_embedding), torch.std(query_embedding)
        query_rand_embedding = torch.randn_like(query_embedding) * q_std + q_mean

        task_embedding = self.task_encoder(task_embedding)
        task_embedding = F.normalize(task_embedding, p=2, dim=-1)

        task_scores = query_rand_embedding @ task_embedding.t()
        task_scores = torch.softmax(task_scores, dim=-1)  # [1, num_tasks]
        return query_rand_embedding, task_scores


class RoleSelector(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, query_embedding, role_embedding):
        query_embedding = query_embedding.expand(role_embedding.size(0), -1)
        role_query_embedding = torch.cat([role_embedding, query_embedding], dim=-1)
        scores = self.node_encoder(role_query_embedding)
        scores = torch.sigmoid(scores).reshape(-1)  # [num_roles]
        selected_role_index = torch.bernoulli(scores).to(torch.bool)
        # 计算选中角色的对数概率
        if selected_role_index.sum() > 0:
            log_probs = torch.log(scores[selected_role_index] + CONST_EPS).sum()
        else:
            # 当没有选择任何角色时，其概率为所有 1 - 选中概率的乘积，在训练时分配其一个固定的较大惩罚值，避免模型不选择任何角色
            log_probs = torch.log(1 - scores + CONST_EPS).sum()
        return selected_role_index, log_probs

    @staticmethod
    def get_selected_role(roles, selected_role_index):
        selected_roles = []
        for i, role in enumerate(roles):
            if selected_role_index[i].item():
                selected_roles.append(role)
        return selected_roles


class StrategySelector(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.role_query_encoder = nn.Linear(in_dim * 2, hidden_dim)
        self.role_role_query_encoder = nn.Linear(in_dim * 3, hidden_dim)
        self.loop_strategy_encoder = nn.Linear(in_dim, hidden_dim)
        self.edge_strategy_encoder = nn.Linear(in_dim, hidden_dim)
        self.loop_predictor = nn.Linear(hidden_dim, 1)
        self.edge_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, query_embedding, selected_role_embedding, edge_strategy_embedding, loop_strategy_embedding):
        num_selected_roles = selected_role_embedding.size(0)

        loop_strategy_embedding = self.loop_strategy_encoder(loop_strategy_embedding)
        loop_strategy_embedding = F.normalize(loop_strategy_embedding, p=2, dim=-1)
        edge_strategy_embedding = self.edge_strategy_encoder(edge_strategy_embedding)
        edge_strategy_embedding = F.normalize(edge_strategy_embedding, p=2, dim=-1)

        selected_edge_tuple = []  # [(source, target, strategy)]
        log_probs = torch.zeros([1], device=query_embedding.device)

        # self-loop strategy
        for i in range(num_selected_roles):
            rq_embedding = torch.cat([selected_role_embedding[i].unsqueeze(0), query_embedding], dim=-1)
            rq_embedding = self.role_query_encoder(rq_embedding)  # [1, hidden_dim]
            rq_embedding = F.normalize(rq_embedding, p=2, dim=-1)
            loop_score = self.loop_predictor(rq_embedding)[0]  # [1,]
            loop_score = torch.sigmoid(loop_score)  # [1,]
            loop_decision = torch.bernoulli(loop_score).to(torch.bool).item()  # scalar
            if loop_decision:
                loop_strategy_score = rq_embedding @ loop_strategy_embedding.t()  # [1, num_loop_strategies]
                loop_strategy_score = torch.softmax(loop_strategy_score, dim=-1)[0]  # [num_loop_strategies, ]
                loop_strategy_index = torch.multinomial(loop_strategy_score, num_samples=1).item()  # scalar
                selected_edge_tuple.append((i, i, loop_strategy_index))
                # 存在自环的 log probability 和 选中的 loop strategy 的 log probability
                log_probs = log_probs + torch.log(loop_score) + torch.log(
                    loop_strategy_score[loop_strategy_index] + 1e-5)
            else:
                # 计算不存在自环的 log probability
                log_probs = log_probs + torch.log(1 - loop_score + 1e-5)

        # edge strategy
        for i in range(num_selected_roles):
            for j in range(num_selected_roles):
                if i == j:
                    continue
                rrq_embedding = torch.cat([
                    selected_role_embedding[i].unsqueeze(0), selected_role_embedding[j].unsqueeze(0),
                    query_embedding], dim=-1)
                rrq_embedding = self.role_role_query_encoder(rrq_embedding)  # [1, hidden_dim]
                rrq_embedding = F.normalize(rrq_embedding, p=2, dim=-1)
                edge_score = self.edge_predictor(rrq_embedding)[0]  # [1,]
                edge_score = torch.sigmoid(edge_score)  # [1,]
                edge_decision = torch.bernoulli(edge_score).to(torch.bool).item()  # scalar
                if edge_decision:
                    edge_strategy_score = rrq_embedding @ edge_strategy_embedding.t()  # [1, num_edge_strategies]
                    edge_strategy_score = torch.softmax(edge_strategy_score, dim=-1)[0]  # [num_edge_strategies, ]
                    edge_strategy_index = torch.multinomial(edge_strategy_score, num_samples=1).item()  # scalar
                    # check if the edge will cause a cycle
                    new_edge = (i, j, edge_strategy_index)
                    if not has_cycle_directed(selected_edge_tuple + [new_edge], drop_self_loops=True):
                        selected_edge_tuple.append(new_edge)
                        # 存在边的 log probability 和 选中的 edge strategy 的 log probability
                        log_probs = log_probs + torch.log(edge_score) + torch.log(
                            edge_strategy_score[edge_strategy_index] + 1e-5)
                else:
                    # 计算不存在边的 log probability
                    log_probs = log_probs + torch.log(1 - edge_score + 1e-5)

        return selected_edge_tuple, log_probs

    @staticmethod
    def get_selected_embedding(edge_strategy_embedding, loop_strategy_embedding, selected_edge_tuple):
        if len(selected_edge_tuple) == 0:
            return None, None

        selected_edge_index = []
        selected_edge_embedding = []
        for src, dst, strategy in selected_edge_tuple:
            selected_edge_index.append([src, dst])
            if src == dst:
                selected_edge_embedding.append(loop_strategy_embedding[strategy])
            else:
                selected_edge_embedding.append(edge_strategy_embedding[strategy])

        selected_edge_index = torch.tensor(
            selected_edge_index, dtype=torch.long, device=edge_strategy_embedding.device).t().contiguous()
        selected_edge_embedding = torch.stack(selected_edge_embedding, dim=0)
        return selected_edge_index, selected_edge_embedding

    @staticmethod
    def get_selected_edges(selected_roles, edge_strategies, loop_strategies, selected_edge_tuple):
        selected_edges = []
        for src, dst, strategy in selected_edge_tuple:
            if src == dst:
                selected_edges.append((selected_roles[src], selected_roles[dst], loop_strategies[strategy]))
            else:
                selected_edges.append((selected_roles[src], selected_roles[dst], edge_strategies[strategy]))
        return selected_edges


class LLMSelector(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.llm_encoder = nn.Linear(in_dim, hidden_dim)
        self.edge_weight_encoder = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.ReLU()
        )
        self.rqe_encoder = GCNConv(in_dim * 2, hidden_dim)

    def forward(self, query_embedding, selected_role_embedding, selected_edge_index,
                selected_edge_embedding, llm_embedding):
        num_selected_roles = selected_role_embedding.size(0)
        llm_embedding = self.llm_encoder(llm_embedding)  # [num_llms, hidden_dim]
        llm_embedding = F.normalize(llm_embedding, p=2, dim=-1)
        query_embedding = query_embedding.expand(num_selected_roles, -1)
        rq_embedding = torch.cat([selected_role_embedding, query_embedding], dim=-1)
        if selected_edge_index is not None and selected_edge_embedding is not None:
            # [num_selected_edges, 1]
            selected_edge_embedding = self.edge_weight_encoder(selected_edge_embedding)
        else:
            # [2, num_selected_roles]
            selected_edge_index = torch.arange(
                num_selected_roles, dtype=torch.long, device=query_embedding.device).unsqueeze(0).repeat(2, 1)
            # Add edge weights for consistency
            selected_edge_embedding = torch.ones(num_selected_roles, device=query_embedding.device)
        # [num_selected_roles, hidden_dim]
        rqe_embedding = self.rqe_encoder(rq_embedding, selected_edge_index, edge_weight=selected_edge_embedding)
        rqe_embedding = F.normalize(rqe_embedding, p=2, dim=-1)

        scores = rqe_embedding @ llm_embedding.t()  # [num_selected_roles, num_llms]
        scores = torch.softmax(scores, dim=-1)  # [num_selected_roles, num_llms]
        selected_llm_index = torch.multinomial(scores, num_samples=1).reshape(-1)  # [num_selected_roles, ]

        # 计算选中的 llm 的 log probability
        log_probs = torch.log(
            scores[torch.arange(num_selected_roles), selected_llm_index] + 1e-5
        ).sum().unsqueeze(0)
        return selected_llm_index, log_probs, selected_edge_index, selected_edge_embedding

    @staticmethod
    def get_selected_llms(llms, selected_llm_index):
        selected_llms = []
        for index in selected_llm_index:
            selected_llms.append(llms[index.item()])
        return selected_llms


class CriticModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.rql_encoder = nn.Linear(embed_dim * 3, embed_dim)
        self.critic_gcn = GCNConv(embed_dim, hidden_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, query_embedding, selected_role_embedding, selected_llm_embedding,
                selected_edge_index, selected_edge_embedding):
        num_selected_roles = selected_role_embedding.size(0)
        # [num_selected_roles, embed_dim * 3]
        rql_embedding = torch.cat([
            selected_role_embedding, selected_llm_embedding, query_embedding.expand(num_selected_roles, -1)
        ], dim=-1)
        rql_embedding = self.rql_encoder(rql_embedding)  # [num_selected_roles, embed_dim]
        rql_embedding = F.normalize(rql_embedding, p=2, dim=-1)  # [num_selected_roles, embed_dim]
        # [num_selected_roles, hidden_dim]
        critic_g_out = self.critic_gcn(rql_embedding, selected_edge_index, edge_weight=selected_edge_embedding)
        # [1, hidden_dim]
        critic_g_out = critic_g_out.mean(dim=0, keepdim=True)
        # [1]
        state_value = self.critic_head(critic_g_out).squeeze(-1)
        return state_value
