# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm import LLM, EngineArgs, LLMEngine, SamplingParams
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


def do_sample(engine: LLMEngine, prompts):
    request_id = 0
    results = []
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params, control_vector_request = prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               control_vector_request=control_vector_request)
            request_id += 1

        request_outputs = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                results.append({
                    "request_id": request_output.request_id,
                    "generation": request_output.outputs[0].text,
                })
    return results


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_control_vector_adapter(requests, enforce_eager):
    # envs.set_vllm_use_v1(use_v1=False)
    engine_args = EngineArgs(
        model=MODEL_PATH,
        enable_control_vector=True,
        max_control_vectors=10,
        max_num_seqs=20,
        gpu_memory_utilization=0.4,
        enforce_eager=enforce_eager,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    result = do_sample(engine, requests)
    print("step result:", result)
    assert len(result) == 10

    del engine


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_offline_inferance(requests, enforce_eager):
    # envs.set_vllm_use_v1(use_v1=False)
    llm = LLM(
        model=MODEL_PATH,
        enable_control_vector=True,
        max_control_vectors=10,
        max_num_seqs=20,
        gpu_memory_utilization=0.4,
        enforce_eager=enforce_eager,
    )

    results = []
    for request in requests:
        prompt, sampling_param, control_vector_request = request
        result = llm.generate(prompt,
                              sampling_param,
                              control_vector_request=control_vector_request)
        results.append({
            "prompt":
            prompt,
            "generation":
            result[0].outputs[0].text,
            "control_vector_name":
            control_vector_request.control_vector_name
            if control_vector_request else None,
            "scale":
            control_vector_request.scale_factor
            if control_vector_request else None
        })
    assert len(results) == 10
    print("generate results:", results)

    del llm
