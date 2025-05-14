from space.llms.base_llm import BaseLM
from groq import Groq, AsyncGroq, BadRequestError
from tenacity import retry, stop_after_attempt, wait_random, retry_if_not_exception_type
from functools import partial
from typing import List, Dict


class GroqChat(BaseLM):
    support = "groq"

    def __init__(self, api_key: str, base_url: str, model: str, record_method="token",
                 prompt_price: float = None, completion_price: float = None, **kwargs):
        super().__init__(model=model, record_method=record_method,
                         prompt_price=prompt_price, completion_price=completion_price)

        self.client = Groq(api_key=api_key, base_url=base_url)
        self.aclient = AsyncGroq(api_key=api_key, base_url=base_url)

        self.normal_chat = partial(self.client.chat.completions.create, model=model, **kwargs)
        self.anormal_chat = partial(self.aclient.chat.completions.create, model=model, **kwargs)

    @retry(wait=wait_random(min=1, max=12), stop=stop_after_attempt(7), reraise=True,
           retry=retry_if_not_exception_type(BadRequestError),
           after=lambda retry_state: print(
               f"`ChatGPT.query` function retried {retry_state.attempt_number} times, waited {retry_state.upcoming_sleep}s"))
    def query(self, messages: List[Dict[str, str]], **kwargs):
        completion = self.normal_chat(messages=messages, **kwargs)
        response = completion.choices[0].message.content
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        self.recorder(
            messages=messages, response=response, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, model=self.model,
            prompt_price=self.prompt_price, completion_price=self.completion_price
        )
        return response

    @retry(wait=wait_random(min=1, max=12), stop=stop_after_attempt(7), reraise=True,
           retry=retry_if_not_exception_type(BadRequestError),
           after=lambda retry_state: print(
               f"`ChatGPT.query` function retried {retry_state.attempt_number} times, waited {retry_state.upcoming_sleep}s"))
    async def aquery(self, messages: List[Dict[str, str]], **kwargs):
        completion = await self.anormal_chat(messages=messages, **kwargs)
        response = completion.choices[0].message.content
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        self.recorder(
            messages=messages, response=response, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, model=self.model,
            prompt_price=self.prompt_price, completion_price=self.completion_price
        )
        return response
