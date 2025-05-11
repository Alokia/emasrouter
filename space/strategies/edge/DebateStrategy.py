from space.strategies.base_strategy import BaseStrategy
from typing import List, Tuple, Dict, Literal
from utils.print_utils import print_with_color


class DebateStrategy(BaseStrategy):
    strategy = "Debate"
    description = "Debate between the two Agents for more in-depth discussion of task."
    self_loop = False
    level = 4

    def __init__(self, num_debate: int = 2, **kwargs):
        super().__init__()
        self.num_debate = num_debate

    def create_strategy_message(self, messages, role: Literal["src", "tgt"],
                                src_response: str = None, has_extra: bool = True):
        n_steps = len(messages)
        smsg = ""

        if role == "src":
            for i in range(n_steps):
                src_msg, tgt_msg = messages[i][0], messages[i][1]
                smsg += f"\n<<Debate Round {i + 1}>>\n"
                smsg += f"\nYour response: \n{src_msg}\n"
                smsg += f"\n**{self.tgt.role}** debates on your response: \n{tgt_msg}\n"
            smsg += f"\n<<Debate Round {n_steps + 1}>>\n" if has_extra else ""
            if n_steps == 0:
                smsg += f"\nGive your first response to start the debate: \n" if has_extra else ""
            else:
                smsg += f"\nGive your response to the debate: \n" if has_extra else ""
        elif role == "tgt":
            for i in range(n_steps):
                src_msg, tgt_msg = messages[i][0], messages[i][1]
                smsg += f"\n<<Debate Round {i + 1}>>\n"
                if i == 0:
                    smsg += f"\n**{self.src.role}** presents a response: \n{src_msg}\n"
                else:
                    smsg += f"\n**{self.src.role}** debates on your response: \n{src_msg}\n"
                smsg += f"\nYour response: \n{tgt_msg}\n" if has_extra else ""
            if src_response is not None:
                smsg += f"\n<<Debate Round {n_steps + 1}>>\n"
                if n_steps == 0:
                    smsg += f"\n**{self.src.role}** presents a response: \n{src_response}\n"
                else:
                    smsg += f"\n**{self.src.role}** debates on your response: \n{src_response}\n"
            smsg += f"\nGive your response to the debate from **{self.src.role}**: \n" if has_extra else ""

        return smsg

    def strategy_prompt(self, role: Literal["src", "tgt"], final: bool = False, **kwargs) -> str:
        prompt = "You are debating with {}, whose occupation is described below: {}\n".format(
            self.src.role if role == "src" else self.tgt.role,
            self.src.description if role == "src" else self.tgt.description
        )
        prompt += "You need to address the task given and debate with him to better solve the task."
        if final:
            prompt += " Next is the final round of debate, where you need to give a final answer to the question based on the history of previous debates."
        else:
            prompt += " Please try your best to give answers that are different or opposite to the other agent."
        return prompt

    def execute(self, query: str, *args, **kwargs) -> Tuple[List[Tuple[str, str]], bool]:
        msg = []
        for i in range(self.num_debate):
            print_with_color(f"    Strategy: Debate. Round: {i + 1}. Executing the {self.src.role}.", color="cyan")
            src_res = self.src.execute(
                query=query,
                strategy_prompt=self.strategy_prompt(role="src", final=i == self.num_debate - 1),
                strategy_message=self.create_strategy_message(msg, role="src")
            )
            print_with_color(f"    Strategy: Debate. Round: {i + 1}. Executing the {self.tgt.role}.", color="cyan")
            tgt_res = self.tgt.execute(
                query=query,
                strategy_prompt=self.strategy_prompt(role="tgt", final=i == self.num_debate - 1),
                strategy_message=self.create_strategy_message(msg, role="tgt", src_response=src_res)
            )
            msg.append((src_res, tgt_res))
        # Store the last round of messages
        self.src.spatial_predecessor_messages[self.tgt.id] = {"role": self.tgt, "output": msg[-1][1]}
        self.tgt.spatial_predecessor_messages[self.src.id] = {"role": self.src, "output": msg[-1][0]}
        return msg, True
