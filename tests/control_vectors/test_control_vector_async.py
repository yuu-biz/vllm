from unittest import result
import pytest

import asyncio
from contextlib import ExitStack
import pytest

from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.control_vectors.request import ControlVectorRequest

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
control_vector_path_happy = \
    "yuu-biz/qwen-cv-example/happy_vector_qwen.gguf"
control_vector_path_spanish = \
    "yuu-biz/qwen-cv-example/english_spanish_vector_qwen.gguf"
spanish = "spanish"

@pytest.fixture
def requests():
    prompt_text = "Write a story about a dog:"  # noqa: E501

    return [
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("spanish", 1, control_vector_path_spanish,
                                 2.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("spanish", 2, control_vector_path_spanish,
                                 1.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            None,
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("spanish", 3, control_vector_path_spanish,
                                 -1.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("spanish", 4, control_vector_path_spanish,
                                 -2.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("happy", 5, control_vector_path_happy, 2.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("happy", 6, control_vector_path_happy, 1.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            None,
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("happy", 7, control_vector_path_happy, -1.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("happy", 8, control_vector_path_happy, -2.0),
        ),
    ]



@pytest.mark.asyncio
@pytest.mark.parametrize("enforce_eager", [False, True])
async def test_control_vector_async(monkeypatch, requests, enforce_eager: bool):
    from vllm.platforms import current_platform
    if not current_platform.is_cuda():
        pytest.skip(reason="V1 currently only supported on CUDA.")

    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        max_control_vectors=10,
        max_num_seqs=20,
        gpu_memory_utilization=0.3,  # Reduce memory usage for tests
        enforce_eager=enforce_eager,
    )

    with monkeypatch.context() as m, ExitStack() as after:
        m.setenv("VLLM_USE_V1", "1")
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        prompts = requests
        request_ids = [f"request-{i}" for i in range(len(prompts))]
        results = []

        # すべてのリクエストを非同期で投げる
        tasks = []
        for idx, (prompt, sampling_params, control_vector_request) in enumerate(prompts):
            tasks.append(
                asyncio.create_task(
                    generate_with_control_vector(
                        engine, request_ids[idx], prompt, sampling_params, control_vector_request
                    )
                )
            )

        # 結果を集める
        done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            result = await task
            results.append(result)

        print("step result:", results)
        assert len(results) == 10
        assert all("request_id" in r and "generation" in r for r in results)

    # Additional cleanup after ExitStack finishes
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

async def generate_with_control_vector(engine, request_id, prompt, sampling_params, control_vector_request):
    # 1リクエスト分の生成を行い、最終出力を返す
    async for output in engine.generate(
        request_id=request_id,
        prompt=prompt,
        sampling_params=sampling_params,
        control_vector_request=control_vector_request
    ):
        if output.finished:
            return {
                "request_id": output.request_id,
                "generation": output.outputs[0].text,
            }
