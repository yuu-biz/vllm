# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import openai
import pytest
import pytest_asyncio
import requests as http_requests

from tests.utils import RemoteOpenAIServer

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
control_vector_path_happy = \
    "yuu-biz/qwen-cv-example/happy_vector_qwen.gguf"
control_vector_path_spanish = \
    "yuu-biz/qwen-cv-example/english_spanish_vector_qwen.gguf"
spanish = "spanish"


@pytest.fixture(scope="session")
def server():
    command = [
        "--served-model-name",
        MODEL_PATH,
        "--port",
        "8000",
        "--enable-control-vector",
        "--max-control-vectors",
        "10000",
        "--control-vectors",
        '{"name": "spanish", "path": "' + control_vector_path_spanish +
        '", "scale_factor": 1.0}',
    ]

    env = {
        "VLLM_ENABLE_CONTROL_VECTOR": "1",
        "VLLM_ALLOW_RUNTIME_CONTROL_VECTOR_UPDATING": "1",
    }

    with RemoteOpenAIServer(model=MODEL_PATH,
                            vllm_serve_args=command,
                            env_dict=env,
                            auto_port=False) as server:
        yield server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_load_control_vector(client: openai.AsyncOpenAI):
    load_control_vector_url = "http://localhost:8000/v1/load_control_vector"

    header = {
        "Content-Type": "application/json",
    }

    params = {
        "control_vector_name": "happy",
        "control_vector_path": control_vector_path_happy,
        "control_vector_scale": 2.0,
    }

    response = http_requests.post(url=load_control_vector_url,
                                  json=params,
                                  headers=header)

    print("Response from server:", response.text)
    assert response.text == \
        "Success: Control vector 'happy' added successfully."


def test_unload_control_vector(client: openai.AsyncOpenAI):
    url = "http://localhost:8000/v1/unload_control_vector"

    header = {
        "Content-Type": "application/json",
    }

    response = http_requests.post(url=url,
                                  json={"control_vector_name": "happy"},
                                  headers=header)

    print("Response from server:", response.text)
    assert response.text == \
        "Success: control vector 'happy' removed successfully."


@pytest.mark.asyncio
async def test_chat_completions_control_vector(client: openai.AsyncOpenAI):

    result = []

    response = await client.chat.completions.create(
        model=spanish,
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": "Write a story about dog:"
        }],
        max_tokens=50,
        temperature=0.0,
        stop=["[/assistant]"],
    )

    result.append(response)
    data = response.to_dict()

    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]
    print("Response from vllm serve:", data)


@pytest.mark.asyncio
async def test_completions_control_vector(client: openai.AsyncOpenAI):
    result = []

    response = await client.completions.create(
        model=spanish,
        prompt="Write a story about a dog:",
        max_tokens=50,
        temperature=0.0,
        stop=["[/assistant]"],
    )

    result.append(response)
    data = response.to_dict()

    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "text" in data["choices"][0]
    print("Response from vllm serve:", data)
