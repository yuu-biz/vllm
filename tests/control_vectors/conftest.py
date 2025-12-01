# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test configuration and fixtures for control vector tests.
"""

import pytest
import torch
import os
import tempfile
from typing import Dict, Any

from vllm.config import ControlVectorConfig
from vllm.control_vectors.request import ControlVectorRequest
from vllm.platforms import current_platform
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)


# Shared control vector file paths for testing
CONTROL_VECTOR_PATH_HAPPY = "yuu-biz/qwen-cv-example/happy_vector_qwen.gguf"
CONTROL_VECTOR_PATH_SPANISH = "yuu-biz/qwen-cv-example/english_spanish_vector_qwen.gguf"


# Global test configuration
def pytest_configure(config):
    """Configure pytest for control vector tests."""
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (deselect with '-m \"not performance\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "skip_global_cleanup: skip global cleanup after test"
    )


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    """Allow subdirectories to skip global cleanup by overriding this fixture."""
    return not request.node.get_closest_marker("skip_global_cleanup")


@pytest.fixture(autouse=True)
def cleanup_fixture(should_do_global_cleanup_after_test: bool):
    yield
    if should_do_global_cleanup_after_test:
        cleanup_dist_env_and_memory(shutdown_ray=True)
        # Additional GPU memory cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


@pytest.fixture
def dist_init():
    """Initialize distributed environment for testing."""
    temp_file = tempfile.mkstemp()[1]

    backend = "nccl"
    if current_platform.is_cpu() or current_platform.is_tpu():
        backend = "gloo"

    init_distributed_environment(world_size=1,
                                 rank=0,
                                 distributed_init_method=f"file://{temp_file}",
                                 local_rank=0,
                                 backend=backend)
    initialize_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture(scope="session")
def device():
    """Session-scoped device fixture."""
    if current_platform.is_cuda_alike() and torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def test_dtype():
    """Session-scoped dtype fixture."""
    return torch.float16


@pytest.fixture
def standard_control_vector_config():
    """Standard control vector configuration for testing."""
    return ControlVectorConfig(
        max_control_vectors=4,
        adapter_dtype=torch.float16,
        normalize=True
    )


@pytest.fixture
def sample_control_vector_request():
    """Sample control vector request for testing."""
    return ControlVectorRequest(
        control_vector_name="test_cv_sample",
        control_vector_id=1,
        control_vector_path=CONTROL_VECTOR_PATH_HAPPY,
        scale_factor=1.0
    )


@pytest.fixture
def multiple_control_vector_requests():
    """Multiple control vector requests for testing."""
    paths = [CONTROL_VECTOR_PATH_HAPPY, CONTROL_VECTOR_PATH_SPANISH, CONTROL_VECTOR_PATH_HAPPY]
    requests = []
    for i in range(3):
        request = ControlVectorRequest(
            control_vector_name=f"test_cv_{i}",
            control_vector_id=i + 1,
            control_vector_path=paths[i],
            scale_factor=1.0 + i * 0.1
        )
        requests.append(request)
    return requests


@pytest.fixture(params=["cpu", "cuda"])
def parametrized_device(request):
    """Parametrized device fixture for cross-device testing."""
    device_type = request.param

    if device_type == "cuda":
        if not current_platform.is_cuda_alike() or not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


@pytest.fixture(params=[torch.float16, torch.float32])
def parametrized_dtype(request):
    """Parametrized dtype fixture for cross-dtype testing."""
    return request.param


@pytest.fixture(params=[1, 2, 4, 8])
def parametrized_max_control_vectors(request):
    """Parametrized max control vectors for capacity testing."""
    return request.param


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)

        # Mark slow tests
        if "long_sequence" in item.nodeid.lower() or "large_batch" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)


# Skip conditions for different environments
def pytest_runtest_setup(item):
    """Setup function run before each test."""

    # Skip CUDA tests if CUDA not available
    if "cuda" in str(item.fixturenames) or "cuda" in item.nodeid.lower():
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    # Skip performance tests in CI unless explicitly requested
    if item.get_closest_marker("performance"):
        if os.getenv("CI") and not os.getenv("RUN_PERFORMANCE_TESTS"):
            pytest.skip("Performance tests skipped in CI")


# Custom assertions for control vectors
def assert_control_vector_request_valid(request: ControlVectorRequest):
    """Assert that a control vector request is valid."""
    assert request.control_vector_name is not None
    assert len(request.control_vector_name) > 0
    assert request.control_vector_id > 0
    assert request.control_vector_path is not None
    assert len(request.control_vector_path) > 0
    assert isinstance(request.scale_factor, (int, float))
    assert not torch.isnan(torch.tensor(request.scale_factor))


def assert_tensor_properties(tensor: torch.Tensor, expected_shape=None, expected_dtype=None, expected_device=None):
    """Assert tensor has expected properties."""
    assert isinstance(tensor, torch.Tensor)

    if expected_shape is not None:
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"

    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"

    if expected_device is not None:
        assert tensor.device.type == expected_device.type, f"Expected device {expected_device}, got {tensor.device}"


# Helper functions for test data
def create_test_tensor(shape, dtype=torch.float16, device=None):
    """Create a test tensor with specified properties."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return torch.randn(shape, dtype=dtype, device=device)


def create_mock_control_vector_data(size: int, dtype=torch.float16, device=None):
    """Create mock control vector data for testing."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return {
        "control_vector": torch.randn(size, dtype=dtype, device=device),
        "layer_names": [f"layer_{i}" for i in range(size // 100)],
        "metadata": {"version": "1.0", "model": "test"}
    }
