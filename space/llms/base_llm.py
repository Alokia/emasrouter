from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Literal, Optional
from class_registry import ClassRegistry
from class_registry.base import AutoRegister
from PIL.Image import Image
from io import BytesIO
import base64
import tiktoken
import warnings

llm_registry = ClassRegistry("support")

llm_price = {
    # https://api-docs.deepseek.com/zh-cn/quick_start/pricing
    "deepseek-chat": {"prompt_price": 2 / 7.24, "completion_price": 8 / 7.24},
    "deepseek-reasoner": {"prompt_price": 4 / 7.24, "completion_price": 16 / 7.24},

    # https://platform.openai.com/docs/pricing
    "gpt-4o-mini": {"prompt_price": 0.15, "completion_price": 0.60},  # gpt-4o-mini-2024-07-18
    "gpt-4o": {"prompt_price": 2.50, "completion_price": 10.00},  # gpt-4o-2024-08-06, gpt-4o-2024-11-20
    "gpt-4o-2024-05-13": {"prompt_price": 5.00, "completion_price": 15.00},
    "chatgpt-4o-latest": {"prompt_price": 5.00, "completion_price": 15.00},
    "gpt-4-turbo": {"prompt_price": 10.00, "completion_price": 30.00},  # gpt-4-turbo-2024-04-09
    "gpt-4": {"prompt_price": 30.00, "completion_price": 60.00},  # gpt-4-0613
    "gpt-4-32k": {"prompt_price": 60.00, "completion_price": 120.00},
    "gpt-3.5-turbo": {"prompt_price": 0.50, "completion_price": 1.50},  # gpt-3.5-turbo-0125
    "gpt-3.5-turbo-instruct": {"prompt_price": 1.50, "completion_price": 2.00},
    "gpt-3.5-turbo-16k-0613": {"prompt_price": 3.00, "completion_price": 4.00},
    "o3-mini": {"prompt_price": 1.10, "completion_price": 4.40},  # o3-mini-2025-01-31
    "o1-mini": {"prompt_price": 1.10, "completion_price": 4.40},  # o1-mini-2024-09-12
    "o1": {"prompt_price": 15.00, "completion_price": 60.00},  # o1-2024-12-17, o1-preview-2024-09-12
    "gpt-4.5-preview": {"prompt_price": 75.00, "completion_price": 150.00},  # gpt-4.5-preview-2025-02-27

    # https://docs.anthropic.com/en/docs/about-claude/models/all-models
    "claude-3-haiku-20240307": {"prompt_price": 0.25, "completion_price": 1.25},
    "claude-3-opus-20240229": {"prompt_price": 15.00, "completion_price": 75.00},
    "claude-3-5-haiku-20241022": {"prompt_price": 0.80, "completion_price": 4.00},
    "claude-3-5-sonnet-20241022": {"prompt_price": 3.00, "completion_price": 15.00},  # claude-3-5-sonnet-20240620
    "claude-3-7-sonnet-20250219": {"prompt_price": 3.00, "completion_price": 15.00},

    # https://ai.google.dev/gemini-api/docs/pricing
    "gemini-2.0-flash": {"prompt_price": 0.10, "completion_price": 0.40},
    "gemini-2.0-flash-lite": {"prompt_price": 0.075, "completion_price": 0.30},
    "gemini-1.5-flash": {"prompt_price": 0.075, "completion_price": 0.30},
    "gemini-1.5-flash-8b": {"prompt_price": 0.0375, "completion_price": 0.15},
    "gemini-1.5-pro": {"prompt_price": 1.25, "completion_price": 5.00},

    # https://groq.com/pricing/
    "llama-3.3-70b-versatile": {"prompt_price": 0.59, "completion_price": 0.79},
    "mixtral-8x7b-32768": {"prompt_price": 0.24, "completion_price": 0.24},
}


class GlobalTokensRecorder:
    def __init__(self, method: Optional[Literal["token", "text"]] = None):
        self.method = method

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_prompt_cost = 0.
        self.total_completion_cost = 0.
        self.total_cost = 0.

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_prompt_cost = 0.
        self.total_completion_cost = 0.
        self.total_cost = 0.

    def _record(self, prompt_tokens, completion_tokens, prompt_cost, completion_cost):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_prompt_cost += prompt_cost
        self.total_completion_cost += completion_cost
        self.total_cost = self.total_prompt_cost + self.total_completion_cost

    @staticmethod
    def get_model_price(model: str) -> Tuple[float, float]:
        if model is not None and model in llm_price:
            prompt_price = llm_price[model]["prompt_price"]
            completion_price = llm_price[model]["completion_price"]
            return prompt_price, completion_price
        warnings.warn("Model not found in llm_price. Returning default values 0. and 0., this will not be recorded.")
        return 0., 0.

    @staticmethod
    def calculate_token(text: str, mode: str = None) -> int:
        if mode is None:
            mode = "gpt-4o"
        encoder = tiktoken.encoding_for_model(mode)
        num_tokens = len(encoder.encode(text))
        return num_tokens

    def calculate_cost_from_text(self, messages: List[Dict[str, str]], response: str,
                                 mode: str = None, model: str = None,
                                 prompt_price: float = None, completion_price: float = None) -> Tuple[float, int, int]:
        if prompt_price is None and completion_price is None:
            prompt_price, completion_price = self.get_model_price(model)

        prompt_tokens = sum([self.calculate_token(msg["content"], mode) for msg in messages])
        completion_tokens = self.calculate_token(response, mode)

        prompt_cost = prompt_tokens * prompt_price / 1000000
        completion_cost = completion_tokens * completion_price / 1000000
        total_cost = prompt_cost + completion_cost
        self._record(prompt_tokens, completion_tokens, prompt_cost, completion_cost)
        return total_cost, prompt_tokens, completion_tokens

    def calculate_cost_from_tokens(self, prompt_tokens: int, completion_tokens: int, prompt_price: float = None,
                                   completion_price: float = None, model: str = None) -> Tuple[float, int, int]:
        if prompt_tokens is None or completion_tokens is None:
            warnings.warn("prompt_tokens and completion_tokens cannot be None. Token Recording will be skipped.")
            return 0., 0, 0
        if prompt_price is None and completion_price is None:
            prompt_price, completion_price = self.get_model_price(model)

        prompt_cost = prompt_tokens * prompt_price / 1000000
        completion_cost = completion_tokens * completion_price / 1000000
        total_cost = prompt_cost + completion_cost
        self._record(prompt_tokens, completion_tokens, prompt_cost, completion_cost)
        return total_cost, prompt_tokens, completion_tokens

    def __call__(self, method: Optional[Literal["token", "text"]] = None,
                 messages: List[Dict[str, str]] = None,
                 response: str = None,
                 mode: str = None,
                 prompt_tokens: int = None,
                 completion_tokens: int = None,
                 model: str = None,
                 prompt_price: float = None,
                 completion_price: float = None) -> Tuple[float, int, int]:
        if method is None:
            method = self.method

        if method == "token":
            return self.calculate_cost_from_tokens(
                prompt_tokens, completion_tokens, prompt_price=prompt_price,
                completion_price=completion_price, model=model
            )
        elif method == "text":
            return self.calculate_cost_from_text(
                messages, response, mode=mode, model=model, prompt_price=prompt_price,
                completion_price=completion_price
            )
        else:
            warnings.warn("method should be 'token' or 'text'. Returning 0. and 0., this will not be recorded.")
            return 0., 0, 0


recorder = GlobalTokensRecorder()


class BaseLM(AutoRegister(llm_registry), ABC):
    support: str = None

    def __init__(self, model: str = None, record_method: Optional[Literal["token", "text"]] = "token",
                 prompt_price: float = None, completion_price: float = None):
        self.model = model
        self.prompt_price = prompt_price
        self.completion_price = completion_price
        self.recorder = recorder
        self.recorder.method = record_method

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model})"

    @staticmethod
    def encode_pil_image_to_base64(image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def create_message(self, text: str, image: Image = None, role: str = "user") -> Dict[str, str]:
        if image is None:
            content = text
        else:
            img_base64 = self.encode_pil_image_to_base64(image)
            content = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ]
        message = {"role": role, "content": content}
        return message

    @abstractmethod
    def query(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass

    @abstractmethod
    async def aquery(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass
