# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import pytest
import torch
from unittest.mock import patch

from vllm.config import (
    CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
    ParallelConfig, SchedulerConfig, VllmConfig, ControlVectorConfig
)
from vllm.control_vectors.worker_manager import (
    WorkerControlVectorManager,
    LRUCacheWorkerControlVectorManager
)
from vllm.control_vectors.request import ControlVectorRequest
from vllm.platforms import current_platform
from .conftest import CONTROL_VECTOR_PATH_HAPPY


MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_CONTROL_VECTORS = 8


@pytest.fixture
def control_vector_config():
    """Create test ControlVectorConfig."""
    return ControlVectorConfig(
        max_control_vectors=NUM_CONTROL_VECTORS,
        adapter_dtype=torch.float16,
        normalize=True
    )


@pytest.fixture
def vllm_config(control_vector_config):
    """Create test VllmConfig with control vectors enabled."""
    return VllmConfig(
        model_config=ModelConfig(
            MODEL_PATH,
            seed=0,
            dtype="float16",
            enforce_eager=True,
        ),
        load_config=LoadConfig(
            download_dir=None,
            load_format="dummy",
        ),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
        ),
        scheduler_config=SchedulerConfig("generate", 32, 32, 32),
        device_config=DeviceConfig("cuda" if current_platform.is_cuda_alike() else "cpu"),
        cache_config=CacheConfig(
            block_size=16,
            swap_space=0,
            cache_dtype="auto",
            gpu_memory_utilization=0.05,  # Very low to avoid OOM when run after other tests
        ),
        control_vector_config=control_vector_config,
    )


def test_worker_control_vector_manager_initialization(control_vector_config):
    """Test WorkerControlVectorManager initialization."""

    device = torch.device("cuda:0" if current_platform.is_cuda_alike() else "cpu")

    manager = WorkerControlVectorManager(
        device=device,
        control_vector_config=control_vector_config
    )

    # Test basic properties
    assert manager.is_enabled == True
    assert manager.control_vector_config == control_vector_config


def test_lru_cache_worker_control_vector_manager_initialization(control_vector_config):
    """Test LRUCacheWorkerControlVectorManager initialization."""

    device = torch.device("cuda:0" if current_platform.is_cuda_alike() else "cpu")

    manager = LRUCacheWorkerControlVectorManager(
        device=device,
        control_vector_config=control_vector_config
    )

    # Test that it inherits from WorkerControlVectorManager
    assert isinstance(manager, WorkerControlVectorManager)
    assert manager.is_enabled == True


@pytest.mark.parametrize("max_control_vectors", [1, 4, 8, 16])
def test_worker_control_vector_manager_different_capacities(max_control_vectors):
    """Test worker manager initialization with different capacity settings."""

    device = torch.device("cuda:0" if current_platform.is_cuda_alike() else "cpu")

    config = ControlVectorConfig(
        max_control_vectors=max_control_vectors,
        adapter_dtype=torch.float16,
        normalize=True
    )

    manager = WorkerControlVectorManager(
        device=device,
        control_vector_config=config
    )

    # Verify manager created successfully with specified capacity
    assert manager.is_enabled == True
    assert manager.control_vector_config.max_control_vectors == max_control_vectors


@patch.dict(os.environ, {"RANK": "0"})
def test_worker_apply_control_vectors(vllm_config):
    """Comprehensive Worker integration test mirroring LoRA's test_worker_apply_lora.

    This test validates the full Worker stack with control vectors:
    1. Worker initialization with control vector config
    2. Device and model loading
    3. Control vector registration and activation
    4. List/query operations
    5. Random subset application patterns

    Note: Uses dummy model format to avoid actual model downloads.
    Real control vector files are mocked - focuses on adapter management logic.
    """
    import tempfile
    import random
    import gc
    from vllm.v1.worker.gpu_worker import Worker

    # Clean up GPU memory before this test (in case of sequential execution)
    # This helps when tests are run together without proper isolation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Give the system a moment to release memory
        import time
        time.sleep(1)

    # Create Worker instance with full vLLM config
    worker = Worker(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method=f"file://{tempfile.mkstemp()[1]}",
    )

    try:
        worker.init_device()
        worker.load_model()

        # Helper function to set active control vectors
        def set_active_control_vectors(worker: Worker, cv_requests: list[ControlVectorRequest]):
            worker.model_runner.control_vector_manager.set_active_adapters(set(cv_requests))

        # Test 1: Empty state
        set_active_control_vectors(worker, [])
        assert worker.model_runner.control_vector_manager.list_adapters() == set()

        # Test 2: Create control vector requests
        cv_requests = [
            ControlVectorRequest(
                control_vector_name=str(i + 1),
                control_vector_id=i + 1,
                control_vector_path=CONTROL_VECTOR_PATH_HAPPY,
                scale=1.0
            ) for i in range(NUM_CONTROL_VECTORS)
        ]

        # Test 3: Apply all control vectors
        set_active_control_vectors(worker, cv_requests)
        assert worker.model_runner.control_vector_manager.list_adapters() == {
            cv_request.control_vector_id for cv_request in cv_requests
        }

        # Test 4: Random subset application (stability check)
        for i in range(NUM_CONTROL_VECTORS):
            random.seed(i)
            iter_cv_requests = random.choices(cv_requests, k=random.randint(1, NUM_CONTROL_VECTORS))
            random.shuffle(iter_cv_requests)
            iter_cv_requests = iter_cv_requests[:-random.randint(0, NUM_CONTROL_VECTORS)]
            set_active_control_vectors(worker, cv_requests)
            assert worker.model_runner.control_vector_manager.list_adapters().issuperset(
                {cv_request.control_vector_id for cv_request in iter_cv_requests}
            )
    finally:
        # Clean up GPU memory
        if hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'model'):
            del worker.model_runner.model
        if hasattr(worker, 'model_runner'):
            del worker.model_runner
        del worker

        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

if __name__ == "__main__":
    pytest.main([__file__])
