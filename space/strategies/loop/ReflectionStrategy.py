from space.strategies.base_strategy import BaseStrategy
from typing import List, Tuple, Dict
from utils.print_utils import print_with_color


class ReflectionStrategy(BaseStrategy):
    strategy = "Reflection"
    description = "Self-loop strategy. Deep reflection on previously given answers."
    self_loop = True
    level = 0

    def __init__(self):
        super().__init__()
        self._count = 0

    def strategy_prompt(self, **kwargs) -> str:
        return (
            "Reflect on possible errors in the answer above and answer again using the same format. "
            "If you think there are no errors in your previous answers that will affect the results, there is no need to correct them."
            "But you must answer again using the same format."
        )

    def execute(
            self, query: str, prompt: List[Dict] = None, response: str = None, *args, **kwargs
    ) -> Tuple[List[Dict[str, str]], bool]:
        # first response will not be used
        if self._count == 0:
            self._count = 1
            return prompt, False
        if self._count == 2:
            self._count = 0
            return [], True

        self._count = 2
        content = "Your previous answer was: " + response + "\n" + self.strategy_prompt()
        msg = prompt + [{'role': 'user', 'content': content}]
        print_with_color(f"    Self-loop Strategy: Reflection. Executing the {self.src.role}.", color="cyan")
        return msg, False
