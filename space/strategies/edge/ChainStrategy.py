from space.strategies.base_strategy import BaseStrategy
from typing import List, Tuple, Dict
from utils.print_utils import print_with_color


class ChainStrategy(BaseStrategy):
    strategy = "Chain"
    description = "Execute the previous node and send its output directly to the next node."
    self_loop = False
    level = 5

    def __init__(self, **kwargs):
        super().__init__()

    def strategy_prompt(self, *args, **kwargs) -> str:
        return ""

    def execute(self, query: str, *args, **kwargs) -> Tuple[List[Tuple[str, str]], bool]:
        print_with_color(f"    Strategy: Chain. Executing the {self.src.role}.", color="cyan")
        msg = self.src.execute(query=query, strategy_prompt=self.strategy_prompt(), strategy_message="")
        self.tgt.spatial_predecessor_messages[self.src.id] = {"role": self.src, "output": msg}
        return [(msg, "")], True
