from space.strategies.base_strategy import BaseStrategy
from typing import List, Tuple, Dict
from utils.print_utils import print_with_color


class CoTStrategy(BaseStrategy):
    strategy = "CoT"
    description = "Self-loop strategy. Using a chain-of-thoughts approach to think about the answer step-by-step"
    self_loop = True
    level = 0

    def __init__(self, num_thoughts: int = 3):
        super().__init__()
        self.num_thoughts = num_thoughts
        assert self.num_thoughts >= 2, "num_thoughts must be greater than or equal to 2"
        self._count = 0
        self._response = []

    def get_response(self):
        msg = ""
        for i in range(len(self._response)):
            msg += "\nThought step" + str(i + 1) + ": \n" + self._response[i] + "\n"
        return msg

    def strategy_prompt(self, **kwargs) -> str:
        return (
            "Given the question, solve it step by step. "
            "Answer your thoughts about the next step of the solution given everything that has been provided to you so far. "
            "Expand on the next step. "
            "Do not try to provide the answer straight away, instead expand on your thoughts about the next step of the solution."
            "Answer in maximum 30 words. "
            "Do not expect additional input. "
            "Make best use of whatever knowledge you have been already provided."
        )

    def execute(
            self, query: str, prompt: List[Dict] = None, response: str = None, *args, **kwargs
    ) -> Tuple[List[Dict[str, str]], bool]:
        if response is not None:
            self._response.append(response)

        self._count += 1
        if self._count > self.num_thoughts:
            self._count = 0
            self._response = []
            return [], True  # stop the loop

        content = self.strategy_prompt() + "\n" + self.get_response()
        if self._count == self.num_thoughts:
            content += "\nThis is the final step, giving the final answer based on the above responses."
        else:
            content += "\nGiving the next step based on the above responses."
        msg = prompt + [{'role': 'user', 'content': content}]
        print_with_color(f"    Self-loop Strategy: CoT. Round {self._count}. Executing the {self.src.role}.", color="cyan")

        return msg, False
