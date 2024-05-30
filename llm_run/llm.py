import os
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from huggingface_hub import InferenceClient
from loguru import logger
from openai import OpenAI

from .utils import get_available_gpus

# fmt: off
try:
    from vllm import LLM, SamplingParams
    VLLM_INSTALLED = True
except Exception as e:
    logger.warning(e)
    VLLM_INSTALLED = False
# fmt: on

PARAMS = {
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 100,
    "max_tokens": 2048,
    "batch_size": 32,
    "repetition_penalty": 1.0,
}


class LLMBase:
    params = PARAMS

    def update_params(self, params=None):
        if isinstance(params, dict):
            self.params.update(params)
        else:
            logger.warning(f"{type(params) = }")

    def load_params(self, params=None):
        _params = self.params.copy()
        if isinstance(params, dict):
            _params.update(params)
        return _params

    @abstractmethod
    def _generate(self, prompt, params) -> str:
        pass

    @abstractmethod
    def _batch_generate(self, prompts, params) -> List:
        pass

    def generate(
        self, prompt: str | List[str], params: Optional[Dict] = None
    ) -> str | List[str]:
        if isinstance(prompt, str):
            return self._generate(prompt, params)
        elif isinstance(prompt, list):
            return self._batch_generate(prompt, params)
        else:
            raise ValueError(f"Check prompt type: {type(prompt)}")


class vllmEngine(LLMBase):
    def __init__(
        self,
        model,
        dtype="float16",
        swap_space=32,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=None,
        **kwargs,
    ):
        assert VLLM_INSTALLED, "install vllm"
        if not tensor_parallel_size:
            tensor_parallel_size = get_available_gpus()
        self.llm = LLM(
            model=model,
            dtype=dtype,
            swap_space=swap_space,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.update_params(kwargs)

    def _generate(self, prompt, params):
        params = self.load_params(params)
        do_sample = params["do_sample"]

        sampling_params = SamplingParams(
            temperature=params["temperature"] if do_sample else 0,
            top_k=params["top_k"] if do_sample else -1,
            top_p=params["top_p"] if do_sample else 1.0,
            max_tokens=params["max_tokens"],
            repetition_penalty=params["repetition_penalty"],
        )
        req_outputs = self.llm.generate(prompt, sampling_params, use_tqdm=False)
        if isinstance(prompt, str):
            return req_outputs[0].outputs[0].text
        else:
            return [r.outputs[0].text for r in req_outputs]

    def _batch_generate(self, prompts, params):
        return self._generate(prompts, params)


class TGIClient(LLMBase):
    def __init__(self, endpoint, params={}):
        self.client = InferenceClient(model=endpoint)
        self.update_params(params)

    def _generate(self, prompt, params):
        params = self.load_params(params)
        do_sample = params["do_sample"]
        return self.client.text_generation(
            prompt,
            do_sample=do_sample,
            temperature=params["temperature"] if do_sample else None,
            top_k=params["top_k"] if do_sample else None,
            top_p=params["top_p"] if do_sample else None,
            max_new_tokens=params["max_tokens"],
            repetition_penalty=params["repetition_penalty"],
            return_full_text=False,
        )

    def _batch_generate(self, prompts, params):
        params = self.load_params(params)
        args = [(prompt, params) for prompt in prompts]
        with ThreadPoolExecutor(params["batch_size"]) as executor:
            results = list(executor.map(self.generate, *zip(*args)))
        return results


class OpenAIClient(LLMBase):
    def __init__(self, model="gpt-3.5-turbo-0125", params={}):
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        assert openai_api_key, "set export OPENAI_API_KEY=''"

        self.client = OpenAI(api_key=openai_api_key)
        params.update({"model": model})
        self.update_params(params)

    def _generate(self, prompt, params):
        params = self.load_params(params)
        response = self.client.chat.completions.create(
            model=params["model"],
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            top_p=params["top_p"],
        )
        return response.choices[0].message.content

    def _batch_generate(self, prompts, params):
        # TODO: check batch API
        params = self.load_params(params)
        args = [(prompt, params) for prompt in prompts]
        with ThreadPoolExecutor(params["batch_size"]) as executor:
            results = list(executor.map(self.generate, *zip(*args)))
        return results


class DeepSeekClient(OpenAIClient):
    def __init__(self, model="deepseek-chat", params={}):
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        assert deepseek_api_key, "set export DEEPSEEK_API_KEY=''"

        self.client = OpenAI(
            api_key=deepseek_api_key, base_url="https://api.deepseek.com"
        )
        params.update({"model": model})
        self.update_params(params)
