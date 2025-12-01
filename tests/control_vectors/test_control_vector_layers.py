# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch

from vllm.config import ControlVectorConfig
from vllm.control_vectors.layers import (
    ControlVectorMapping,
    MLPWithControlVector,
    BaseLayerWithControlVector
)
from vllm.platforms import current_platform

# Test tolerances for different data types
TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.float32: (1e-4, 1e-4),
    torch.bfloat16: (3e-2, 2e-2),
}

DEVICES = ([
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
] if current_platform.is_cuda_alike() else ["cpu"])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mlp_with_control_vector_initialization(device, dtype):
    """Test MLPWithControlVector initialization."""

    # Create a dummy base layer (MLP)
    base_layer = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512)
    ).to(device=device, dtype=dtype)

    hidden_size = 512

    # Initialize MLPWithControlVector
    mlp_with_cv = MLPWithControlVector(base_layer, hidden_size, dtype)
    mlp_with_cv = mlp_with_cv.to(device=device, dtype=dtype)

    # Test basic properties
    assert mlp_with_cv.hidden_size == hidden_size
    assert mlp_with_cv.dtype == dtype
    assert mlp_with_cv.normalize == True
    assert mlp_with_cv.active_vector.shape == (hidden_size,)
    assert mlp_with_cv.active_vector.dtype == dtype
    assert len(mlp_with_cv.control_vectors) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_control_vector_operations(device, dtype):
    """Test control vector set/get operations."""

    base_layer = torch.nn.Linear(512, 512).to(device=device, dtype=dtype)
    hidden_size = 512

    mlp_with_cv = MLPWithControlVector(base_layer, hidden_size, dtype)
    mlp_with_cv = mlp_with_cv.to(device=device, dtype=dtype)

    # Create test control vectors
    cv1 = torch.randn(hidden_size, dtype=dtype, device=device)
    cv2 = torch.randn(hidden_size, dtype=dtype, device=device)

    # Test setting control vectors
    mlp_with_cv.set_control_vector(1, cv1)
    mlp_with_cv.set_control_vector(2, cv2)

    # Test getting control vectors
    retrieved_cv1 = mlp_with_cv.get_control_vector(1)
    retrieved_cv2 = mlp_with_cv.get_control_vector(2)

    assert torch.allclose(retrieved_cv1, cv1, rtol=1e-4, atol=1e-4)
    assert torch.allclose(retrieved_cv2, cv2, rtol=1e-4, atol=1e-4)

    # Test non-existent index
    assert mlp_with_cv.get_control_vector(999) is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_control_vector_forward_pass(device, dtype):
    """Test forward pass with control vectors."""

    # Skip float16 test on CPU as it's not well supported
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not well supported on CPU")

    batch_size = 4
    hidden_size = 512

    # Create base MLP layer
    base_layer = torch.nn.Linear(hidden_size, hidden_size).to(device=device, dtype=dtype)

    mlp_with_cv = MLPWithControlVector(base_layer, hidden_size, dtype)
    mlp_with_cv = mlp_with_cv.to(device=device, dtype=dtype)

    # Create input tensor
    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

    # Test forward pass without control vector
    output_without_cv = mlp_with_cv(input_tensor)
    assert output_without_cv.shape == (batch_size, hidden_size)

    # Add a control vector and test forward pass
    control_vector = torch.randn(hidden_size, dtype=dtype, device=device) * 0.1
    mlp_with_cv.set_control_vector(1, control_vector)
    mlp_with_cv.set_active_tensor(1)

    output_with_cv = mlp_with_cv(input_tensor)
    assert output_with_cv.shape == (batch_size, hidden_size)

    # Output should be different when control vector is applied
    assert not torch.allclose(output_without_cv, output_with_cv, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_control_vector_mapping(device):
    """Test ControlVectorMapping functionality."""

    # Create test mapping
    layer_mapping = {
        0: torch.randn(512, device=device),
        5: torch.randn(512, device=device),
        10: torch.randn(512, device=device)
    }

    mapping = ControlVectorMapping(layer_mapping=layer_mapping)

    # Test basic properties
    assert len(mapping.layer_mapping) == 3
    assert 0 in mapping.layer_mapping
    assert 5 in mapping.layer_mapping
    assert 10 in mapping.layer_mapping
    assert 999 not in mapping.layer_mapping

    # Test tensor retrieval
    assert torch.allclose(mapping.layer_mapping[0], layer_mapping[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_control_vector_dtype_handling(dtype):
    """Test different data type handling."""

    if not current_platform.is_cuda_alike() and dtype == torch.float16:
        pytest.skip("float16 not well supported on CPU")

    device = "cuda:0" if current_platform.is_cuda_alike() else "cpu"
    hidden_size = 256

    base_layer = torch.nn.Linear(hidden_size, hidden_size).to(device=device, dtype=dtype)

    mlp_with_cv = MLPWithControlVector(base_layer, hidden_size, dtype)
    mlp_with_cv = mlp_with_cv.to(device=device, dtype=dtype)

    # Test that internal tensors have correct dtype
    assert mlp_with_cv.active_vector.dtype == dtype

    # Add control vector and check dtype consistency
    control_vector = torch.randn(hidden_size, dtype=dtype, device=device)
    mlp_with_cv.set_control_vector(1, control_vector)

    retrieved_cv = mlp_with_cv.get_control_vector(1)
    assert retrieved_cv.dtype == dtype


def test_control_vector_normalization():
    """Test control vector normalization functionality."""

    device = "cuda:0" if current_platform.is_cuda_alike() else "cpu"
    hidden_size = 128

    base_layer = torch.nn.Linear(hidden_size, hidden_size).to(device=device)

    mlp_with_cv = MLPWithControlVector(base_layer, hidden_size, torch.float32)
    mlp_with_cv = mlp_with_cv.to(device=device)

    # Test normalization setting
    assert mlp_with_cv.normalize == True

    mlp_with_cv.set_normalization(False)
    assert mlp_with_cv.normalize == False

    mlp_with_cv.set_normalization(True)
    assert mlp_with_cv.normalize == True


@pytest.mark.parametrize("device", DEVICES)
def test_layer_id_functionality(device):
    """Test layer ID assignment functionality."""

    base_layer = torch.nn.Linear(256, 256).to(device=device)

    mlp_with_cv = MLPWithControlVector(base_layer, 256, torch.float32)
    mlp_with_cv = mlp_with_cv.to(device=device)

    # Test layer ID setting
    layer_id = 42
    mlp_with_cv.set_layer_id(layer_id)
    assert mlp_with_cv.layer_id == layer_id


def test_base_layer_with_control_vector():
    """Test BaseLayerWithControlVector base class."""

    # Test that it can be instantiated (though it's mostly abstract)
    base_layer = BaseLayerWithControlVector()
    assert isinstance(base_layer, torch.nn.Module)


if __name__ == "__main__":
    pytest.main([__file__])
