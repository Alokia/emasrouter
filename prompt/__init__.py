from .base_prompt import prompt_registry


def get_prompt(domain: str):
    return prompt_registry.get(domain)
