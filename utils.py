import re

def remove_think_tags(text: str) -> str:
    """
    Strips out any <think>...</think> sections in the LLM output.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()