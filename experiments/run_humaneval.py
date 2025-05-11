from models.gem import ACGEM
from models.loss_func import LossFunction
from space.llms import llm_pool
from space.strategies import edge_strategy_pool, loop_strategy_pool
from space.roles import role_pool
from space.task import task_pool
from dataset.human_eval_dataset import HumanEvalDataset, HumanEvalDataLoader
from graph.base_graph import BaseGraph
from graph.graph_visualization_mermaid import show_mermaid_graph
from utils.graph_utils import make_mermaid
from utils.print_utils import print_with_color
from tools.coding.python_executor import PyExecutor
from space.llms.base_llm import recorder
from loguru import logger
import os
import argparse
import torch
import re
import math

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger.add("../logs/human_eval/train.log")


def get_str(arr, is_tuple=False):
    if not is_tuple:
        str_arr = [a["name"] for a in arr]
        return "\n".join(str_arr)
    else:
        str_arr = [a["name"] + " - " + b["name"] + " - " + c["name"] for a, b, c in arr]
        return "\n".join(str_arr)


def train_step(model, optimizer, train_loader, loss_func, epoch):
    print_with_color(f"------ Training epoch {epoch} ------", "red")
    pattern = r'```python.*```'
    total_solved, total_executed = (0, 0)
    prompt_tokens, completion_tokens, cost = 0, 0, 0.

    steps = 0
    for data in train_loader:
        steps += 1
        is_solved = 0

        data = data[0]
        query = data["prompt"]
        test = data["test"]
        entry_point = data["entry_point"]
        task_label = torch.tensor([2], dtype=torch.long).to(model.device)

        logger.debug(
            f"\n>> Epoch: {epoch} ---- Training step: {steps}\n"
            f"\n>> Query:\n{query}\n" +
            f"\n>> Entry point:\n{entry_point}\n" +
            f"\n>> Test:\n{test}\n"
        )

        optimizer.zero_grad()
        flag = False
        while not flag:
            task_scores, selected_roles, selected_edges, selected_llms, log_probs, state_value = model(
                query=query,
                tasks=task_pool,
                roles=role_pool["code"],
                edge_strategies=edge_strategy_pool, loop_strategies=loop_strategy_pool,
                llms=llm_pool
            )

            logger.debug(
                f"\n-- Selected Roles:\n{get_str(selected_roles)}\n" +
                f"\n-- Selected Edges:\n{get_str(selected_edges, is_tuple=True)}\n" +
                f"\n-- Selected LLMs:\n{get_str(selected_llms)}\n"
            )

            flag, loss = loss_func.pre_calculate(log_probs, selected_roles, selected_edges, selected_llms)

            if not flag:
                logger.debug(f"\nInvalid graph, retrying...\n loss: {loss.item()}")
            else:
                recorder.reset()

                graph = BaseGraph(domain="code", dataset="human_eval")
                graph.create_graph(selected_roles, selected_edges, selected_llms)
                mermaid = make_mermaid(
                    [r["instance"] for r in selected_roles],
                    [(s["instance"], t["instance"], st["instance"]) for s, t, st in selected_edges]
                )
                show_mermaid_graph(mermaid)
                result = graph.run(query)

                logger.debug(f"\n** Graph Result:\n{result}")

                prompt_tokens += recorder.total_prompt_tokens
                completion_tokens += recorder.total_completion_tokens
                cost += recorder.total_cost

                logger.info(
                    f"\n== Prompt Tokens: {recorder.total_prompt_tokens}" +
                    f"\n== Completion Tokens: {recorder.total_completion_tokens}" +
                    f"\n== Cost: {recorder.total_cost}" +
                    f"\n== Current total prompt tokens: {prompt_tokens}, Current total completion tokens: {completion_tokens}, Current total cost: {cost}"
                )

                match = re.search(pattern, result, re.DOTALL | re.MULTILINE)
                if match:
                    answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                    is_solved = PyExecutor().evaluate(entry_point, answer, test, timeout=100)

                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                loss = loss_func(
                    True, log_probs=log_probs, total_cost=recorder.total_cost,
                    is_solved=is_solved, state_value=state_value, num_selected_roles=len(selected_roles),
                )

            loss = loss + loss_func.calculate_task_loss(task_scores, task_label)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"../save/human_eval_step_new.pth")
        logger.info(
            f"\n%% Epoch: {epoch} -- Training step: {steps}" +
            f"\n%% is_solved: {is_solved}" +
            f"\n%% Loss: {loss.item()}" +
            f"\n%% Accuracy: {total_solved / total_executed}" +
            f"\n%% Total Solved: {total_solved}" +
            f"\n%% Total Executed: {total_executed}"
        )

    return prompt_tokens, completion_tokens, cost


@torch.no_grad()
def inference(loader, model, loss_func):
    model.eval()
    pattern = r'```python.*```'
    total_solved, total_executed = (0, 0)
    prompt_tokens, completion_tokens, cost = 0, 0, 0.

    steps = 0
    for data in loader:
        steps += 1

        data = data[0]
        query = data["prompt"]
        test = data["test"]
        entry_point = data["entry_point"]

        logger.debug(
            f"\n>> Inference step: {steps}\n"
            f"\n>> Query:\n{query}\n" +
            f"\n>> Entry point:\n{entry_point}\n" +
            f"\n>> Test:\n{test}\n"
        )

        flag = False
        while not flag:
            task_scores, selected_roles, selected_edges, selected_llms, log_probs, state_value = model(
                query=query,
                tasks=task_pool,
                roles=role_pool["code"],
                edge_strategies=edge_strategy_pool, loop_strategies=loop_strategy_pool,
                llms=llm_pool
            )

            logger.debug(
                f"\n-- Selected Roles:\n{get_str(selected_roles)}\n" +
                f"\n-- Selected Edges:\n{get_str(selected_edges, is_tuple=True)}\n" +
                f"\n-- Selected LLMs:\n{get_str(selected_llms)}\n"
            )

            flag, loss = loss_func.pre_calculate(log_probs, selected_roles, selected_edges, selected_llms)
            if not flag:
                logger.debug(f"\nInvalid graph, retrying...")
            else:
                recorder.reset()

                graph = BaseGraph(domain="code", dataset="human_eval")
                graph.create_graph(selected_roles, selected_edges, selected_llms)
                mermaid = make_mermaid(
                    [r["instance"] for r in selected_roles],
                    [(s["instance"], t["instance"], st["instance"]) for s, t, st in selected_edges]
                )
                show_mermaid_graph(mermaid)
                result = graph.run(query)

                logger.debug(f"\n** Graph Result:\n{result}")

                prompt_tokens += recorder.total_prompt_tokens
                completion_tokens += recorder.total_completion_tokens
                cost += recorder.total_cost

                logger.info(
                    f"\n== Prompt Tokens: {recorder.total_prompt_tokens}" +
                    f"\n== Completion Tokens: {recorder.total_completion_tokens}" +
                    f"\n== Cost: {recorder.total_cost}" +
                    f"\n== Current total prompt tokens: {prompt_tokens}, Current total completion tokens: {completion_tokens}, Current total cost: {cost}"
                )

                match = re.search(pattern, result, re.DOTALL | re.MULTILINE)
                is_solved = 0
                if match:
                    answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                    is_solved = PyExecutor().evaluate(entry_point, answer, test, timeout=100)

                total_solved = total_solved + is_solved
                total_executed = total_executed + 1

                logger.info(
                    f"\n%% Inference step: {steps}" +
                    f"\n%% is_solved: {is_solved}" +
                    f"\n%% Accuracy: {total_solved / total_executed}" +
                    f"\n%% Total Solved: {total_solved}" +
                    f"\n%% Total Executed: {total_executed}"
                )

    logger.info(
        f"\nInference accuracy: {total_solved / total_executed}" +
        f"\nTotal prompt tokens: {prompt_tokens}" +
        f"\nTotal completion tokens: {completion_tokens}" +
        f"\nTotal cost: {cost}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../dataset/human_eval/HumanEval.jsonl")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--lambda_cost", type=float, default=5.0)
    parser.add_argument("--inference", action="store_true", default=False, help="whether to run inference")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_dataset, test_dataset = HumanEvalDataset.split(path=args.dataset_path, ratio=[0.2, 0.8], shuffle=True)
    train_loader = HumanEvalDataLoader(train_dataset)
    test_loader = HumanEvalDataLoader(test_dataset)
    loss_func = LossFunction(lambda_cost=args.lambda_cost, logger=logger)

    device = torch.device("cuda") if args.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")

    model = ACGEM(embed_dim=384, hidden_dim=64, device=device, latent_dim=64).to(device)
    if args.load_ckpt is not None:
        logger.debug(f"\nLoading checkpoint from {args.load_ckpt}")
        model.load_state_dict(torch.load(args.load_ckpt))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_prompt_tokens, total_completion_tokens, total_cost = 0, 0, 0.

    for epoch in range(args.start_epoch, args.epochs):
        logger.debug(f"\n\n********************************************************************\n\n Epoch: {epoch}")
        tpt, tct, tc = train_step(model, optimizer, train_loader, loss_func, epoch)
        total_prompt_tokens += tpt
        total_completion_tokens += tct
        total_cost += tc
        logger.info(f"Total Prompt Tokens: {total_prompt_tokens}, Total Completion Tokens: {total_completion_tokens}, Total Cost: {total_cost}")
        torch.save(model.state_dict(), f"../save/human_eval_{epoch}.pth")

    if args.inference:
        inference(test_loader, model)
