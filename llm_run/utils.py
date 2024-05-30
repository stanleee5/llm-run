import time

import pynvml
from loguru import logger


def apply_chat_template(tokenizer, prompt: str) -> str:
    """
    apply chat tempate and remove bos_token
    """
    if not tokenizer:
        return prompt
    chat = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    if tokenizer.bos_token in prompt:
        start_p = prompt.find(tokenizer.bos_token) + len(tokenizer.bos_token)
    else:
        start_p = 0
    return prompt[start_p:]


def get_available_gpus() -> int:
    """
    Returns the number of available GPUs on the system.
    """
    pynvml.nvmlInit()
    try:
        gpu_count = pynvml.nvmlDeviceGetCount()
        return gpu_count
    except Exception as e:
        logger.error(e)
        return 0


def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(
            f"Function {func.__name__} took {end_time - start_time:.3f} seconds to run."
        )
        return result

    return wrapper
