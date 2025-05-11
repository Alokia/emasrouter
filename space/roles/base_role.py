from graph.base_node import BaseNode
from prompt.output_format import output_format_prompt
from prompt.message_aggregation import message_aggregation, inner_test
from prompt.post_process import post_process


class BaseRole(BaseNode):
    role: str = None
    description: str = None
    post_description: str = None

    def __init__(self, role: str = None,
                 description: str = None, pre_aggregation: str = None, pre_output_format: str = None,
                 post_description: str = None, post_aggregation: str = None, post_output_format: str = None):
        super().__init__(role=role)

        self.description = description
        self.pre_aggregation = pre_aggregation
        self.pre_output_format = pre_output_format

        self.post_description = post_description
        self.post_aggregation = post_aggregation
        self.post_output_format = post_output_format

    def __str__(self):
        return self.role

    def __repr__(self):
        return f"{self.__class__.__name__}({self.role})"

    @property
    def name(self):
        return self.role + "-" + self.id

    def _pre_process(self, query: str, strategy_prompt: str, strategy_message: str = "", **kwargs):
        spatial_prompt = message_aggregation({"query": query}, self.spatial_predecessor_messages, self.pre_aggregation)
        format_prompt = output_format_prompt[self.pre_output_format]

        system_prompt = f"{self.description}\n{strategy_prompt}"
        system_prompt += f"\nFormat requirements that must be followed:\n{format_prompt}" if format_prompt else ""
        user_prompt = f"{query}\n"
        user_prompt += f"At the same time, other agents' outputs are as follows:\n\n{spatial_prompt}" if spatial_prompt else ""
        user_prompt += f"\n\n Here is the information you make before:\n\n{strategy_message}" if strategy_message else ""
        return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]

    def _execute(self, query: str, strategy_prompt: str = "", strategy_message: str = "", **kwargs):
        passed, response = inner_test({"query": query}, self.spatial_predecessor_messages, {})
        if passed:
            return response
        prompt = self._pre_process(query, strategy_prompt, strategy_message=strategy_message, **kwargs)
        if self.llm is None:
            raise RuntimeError(f"Role {self.role} is not defined.")
        if self.loop_strategy is not None:
            prompt, flag = self.loop_strategy.execute(query, prompt=prompt)
            while not flag:
                response = self.llm.query(prompt)
                print(f"\n******** {self.role} -- self-loop -- execute: \n" + response + "\n")
                response = self._post_process(query, response, **kwargs)
                print(f"\n******** {self.role} -- self-loop -- post_process: \n" + response + "\n")
                prompt, flag = self.loop_strategy.execute(query, prompt=prompt, response=response)
        else:
            response = self.llm.query(prompt)
            print(f"\n******** {self.role} -- execute: \n" + response + "\n")
            response = self._post_process(query, response, **kwargs)
            print(f"\n******** {self.role} -- post_process: \n" + response + "\n")
        return response

    def _post_process(self, query: str, response: str, **kwargs):
        response = post_process({"query": query}, response, self.post_aggregation)
        post_format_prompt = output_format_prompt[self.post_output_format]
        if post_format_prompt is not None:
            system_prompt = f"{self.post_description}\n"
            system_prompt += f"Format requirements that must be followed:\n{post_format_prompt}"
            user_prompt = f"{query}\nThe initial thinking information is:\n{response} Please refer to the new format requirements when replying."
            prompt = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
            response = self.llm.query(prompt)
        return response



