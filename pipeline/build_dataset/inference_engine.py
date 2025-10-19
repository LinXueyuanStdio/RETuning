
import os
import openai
from loguru import logger

try:
    from vllm import LLM, SamplingParams
except ImportError:
    logger.warning("VLLM not installed. Please install vllm with 'pip install vllm'.")
import random


class InferenceEngine:
    def inference(self, prompts: list[str | list[dict]], n=1, **kwargs) -> list[list[str] | str]:
        raise NotImplementedError("For each prompt, return n responses. If n=1, the return is a list of one element list of strings.")

    def inference_one(self, prompt: str | list[dict] | list[str | list[dict]], **kwargs) -> list[str] | str:
        # This function is used to get one response for each prompt.
        if isinstance(prompt, str):
            prompts = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            if isinstance(prompt[0], dict):
                prompts = [prompt]
            else:
                prompts = prompt
        response = self.inference(prompts, n=1, **kwargs)
        if isinstance(response, list) and len(response) == 1:
            return response[0]
        return response

    def __call__(self, prompts: list[str | list[dict]], n=1, **kwargs) -> list[list[str] | str]:
        response = self.inference(prompts, n=n, **kwargs)
        return response if isinstance(response, list) else [response]



class OpenAIApiInferenceEngine(InferenceEngine):
    def __init__(self, model: str, max_tokens: int, temperature: float, top_p: float = 1.0):
        self.model = model
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def inference(self, prompts: list[str] | list[list[dict]], n=1, **kwargs) -> list[list[str]]:
        responses = []
        for prompt in prompts:
            if not prompt:
                responses.append([None] * n)
                continue
            if isinstance(prompt, dict):
                messages = [prompt]
            elif isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                messages = prompt
            else:
                raise ValueError(f"Invalid prompt format: {prompt}")
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                stop=kwargs.get("stop", None),
                n=n,
                stream=False,
            )
            n_responds = []
            for i in range(n):
                msg = response.choices[i].message
                content = msg["content"] if "content" in msg else ""
                if "reasoning_content" in msg:
                    reasoning_content = msg["reasoning_content"]
                    content = f"<think>\n{reasoning_content}\n</think>\n{content}"
                n_responds.append(content)
            responses.append(n_responds)
        return responses


class vllmInferenceEngine(InferenceEngine):
    def __init__(self,
                 model: str,
                 max_tokens: int,
                 temperature: float,
                 top_p: float,
                 tensor_parallel_size: int,
                 max_model_len: int,
                 gpu_memory_utilization: float = 0.95,
                 **kwargs):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=False,
            **kwargs,
        )

    def inference(self, prompts: list[str] | list[list[dict]], n=1, **kwargs) -> list[list[str]]:
        not_none_index = [i for i, prompt in enumerate(prompts) if prompt is not None]
        # If all prompts are None, return a list of None
        if len(not_none_index) == 0:
            return [[None] * n for _ in range(len(prompts))]
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            n=n,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        responses = []
        for output in outputs:
            n_responses = []
            for i in range(n):
                n_responses.append(output.outputs[i].text)
            responses.append(n_responses)
        # If all prompts are not None, return the responses
        if len(not_none_index) == len(prompts):
            return responses
        # If some prompts are None, return the responses for the non-None prompts and None for the None prompts
        # and keep the order of the original prompts
        final_responses = []
        for i in range(len(prompts)):
            if i in not_none_index:
                final_responses.append(responses[not_none_index.index(i)])
            else:
                final_responses.append([None] * n)
        return final_responses


def build_inference_engine(
    engine: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 1.0,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.95,
    **kwargs,
) -> InferenceEngine:
    """Build inference engine based on the provided arguments."""

    if engine == "openai":
        return OpenAIApiInferenceEngine(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    elif engine == "vllm":
        return vllmInferenceEngine(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown engine: {engine}")
