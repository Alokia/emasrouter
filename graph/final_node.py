from graph.base_node import BaseNode
from typing import Dict, List
from collections import Counter

final_config = {
    "human_eval": {
        "system": "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers. \nUse a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```\nDo not include anything other than Python code blocks in your response.",
        "user": "You will be given a function signature and its docstring by the user.\nYou may be given the overall code design, algorithm framework, code implementation or test problems.\nWrite your full implementation (restate the function signature).\nUse a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```\nDo not include anything other than Python code blocks in your response."
    },
    "gsm8k": {
        "system": "You will be given a math problem, analysis and code from other agents. Please find the most reliable answer based on the analysis and results of other agents. Give reasons for making decisions. The last line of your output contains only the final result without any units, for example: The answer is 140. However, The answer is 140$ or The answer is Option A or The answer is A.140 is not allowed. Remember not to add units and no other content.",
        "user": "Please provide the final answer based on the analysis and results of other agents. For example: the answer is 140 or the answer is 0"
    },
    "math": {
        "system": "You will be given a math problem, analysis and code from other agents. Please find the most reliable answer based on the analysis and results of other agents. Give reasons for making decisions. You answer should be wrapped by \\boxed{}  without any units, for example: The answer is \\boxed{140}. Remember not to add units.",
        "user": "Please provide the final answer based on the analysis and results of other agents. For example: the answer is \\boxed{140} or the answer is \\boxed{140}"
    },
    "mbpp": {
        "system": "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers. \nUse a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```\nDo not include anything other than Python code blocks in your response.",
        "user": "You will be given a function signature and its docstring by the user.\nYou may be given the overall code design, algorithm framework, code implementation or test problems.\nWrite your full implementation (restate the function signature).\nUse a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```\nDo not include anything other than Python code blocks in your response."
    },
    "mmlu": {
        "system": "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers.",
        "user": "\nOnly one answer out of the offered 4 is correct.\nYou must choose the correct answer to the question.\nYour response must be one of the 4 letters: A, B, C or D, corresponding to the correct answer.\nI will give you some other people's answers and analysis.\nThe last line of the reply should contain only one sentence(the answer is \\boxed{A/B/C/D}.) and nothing else.\nFor example, The answer is the answer is \\boxed{A}."
    }
}


class FinalNode(BaseNode):
    def __init__(self, dataset: str):
        super().__init__(role="Final Refer")
        self.dataset = dataset

    def __str__(self):
        return self.role

    def __repr__(self):
        return f"{self.__class__.__name__}({self.role})"

    @property
    def name(self):
        return self.role + "-" + self.id

    def set_llm(self, llms: List[Dict]):
        llm_names = [llm["name"] for llm in llms]
        count = Counter(llm_names)
        mode, _ = count.most_common(1)[0]
        for llm in llms:
            if llm["name"] == mode:
                super().set_llm(llm["instance"])
                break

    def _pre_process(self, query: str, outputs: Dict[str, Dict], **kwargs):
        system_prompt = f"{final_config[self.dataset]['system']}"
        spatial_str = ""
        for node_id, info in outputs.items():
            spatial_str += "Node Id: " + node_id + " Role Name: " + info["role"] + " Output: " + info['output'] + "\n\n"
        user_prompt = f"The task is:\n\n {query}.\n At the same time, the output of other agents is as follows:\n\n{spatial_str} {final_config[self.dataset]['user']}"
        return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]

    def _execute(self, query: str, outputs: Dict[str, Dict], **kwargs):
        prompt = self._pre_process(query, outputs, **kwargs)
        response = self.llm.query(prompt)
        return response

    def _post_process(self, query: str, pre_result, exec_result, **kwargs):
        raise NotImplementedError()
