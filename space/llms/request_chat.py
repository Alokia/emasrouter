from space.llms.base_llm import BaseLM
from tenacity import retry, stop_after_attempt, wait_random, retry_if_not_exception_type
from typing import List, Dict
import requests


class RequestChat(BaseLM):
    support = "request"

    def __init__(self, api_key: str, base_url: str, model: str, record_method="text",
                 prompt_price: float = None, completion_price: float = None, **kwargs):
        super().__init__(model=model, record_method=record_method,
                         prompt_price=prompt_price, completion_price=completion_price)
        self.api_key = api_key
        self.base_url = base_url

    @retry(wait=wait_random(min=1, max=12), stop=stop_after_attempt(7), reraise=True,
           retry=retry_if_not_exception_type(requests.HTTPError),
           after=lambda retry_state: print(
               f"`ChatGPT.query` function retried {retry_state.attempt_number} times, waited {retry_state.upcoming_sleep}s"))
    def query(self, messages: List[Dict[str, str]], **kwargs):
        response = requests.post(self.base_url, json={"messages": messages}, headers={'x-api-key': self.api_key})
        self.recorder(
            messages=messages, response=response, model=self.model,
            prompt_price=self.prompt_price, completion_price=self.completion_price
        )
        return response

    async def aquery(self, messages: List[Dict[str, str]], **kwargs):
        raise NotImplementedError()
