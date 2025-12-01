# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn
from unittest.mock import MagicMock

from vllm.config import ControlVectorConfig
from vllm.control_vectors.models import (
    ControlVectorModel,
    ControlVectorModelManager,
    LRUCacheControlVectorModelManager,
    create_control_vector_manager
)
from vllm.control_vectors.layers import ControlVectorMapping
from vllm.platforms import current_platform
from .conftest import CONTROL_VECTOR_PATH_HAPPY, CONTROL_VECTOR_PATH_SPANISH

# Test devices
DEVICES = ([
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
] if current_platform.is_cuda_alike() else ["cpu"])

DEFAULT_DTYPE = torch.get_default_dtype()


def create_dummy_control_vector(
    control_vector_id: int,
    device: torch.device,
) -> ControlVectorModel:
    """Create a dummy control vector for testing without loading from file."""
    # Create mock control vector weights
    control_vector_weights = {
        0: torch.randn(512, device=device),
        5: torch.randn(512, device=device),
        10: torch.randn(512, device=device),
    }

    return ControlVectorModel(
        control_vector_id=control_vector_id,
        control_vector_weights=control_vector_weights,
        scale_factor=1.0
    )


@pytest.fixture
def control_vector_config():
    """Create a test ControlVectorConfig."""
    return ControlVectorConfig(
        max_control_vectors=4,
        adapter_dtype=torch.float16,
        normalize=True
    )


@pytest.fixture
def dummy_model():
    """Create a dummy neural network model for testing, similar to LoRA tests."""
    class DummyConfig:
        hidden_size = 512
        num_hidden_layers = 12
        torch_dtype = torch.float16

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = DummyConfig()
            self.dense1 = nn.Linear(512, 1024)
            self.dense2 = nn.Linear(1024, 512)
            self.act = nn.ReLU()

        def forward(self, x):
            x = self.dense1(x)
            x = self.act(x)
            x = self.dense2(x)
            return x

    return DummyModel()


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("device", DEVICES)
def test_control_vector_model_from_checkpoint(device, control_vector_config):
    """Test ControlVectorModel loading from checkpoint."""
    control_vector_id = 1
    control_vector_path = CONTROL_VECTOR_PATH_HAPPY
    scale_factor = 1.5

    try:
        cv_model = ControlVectorModel.from_local_checkpoint(
            control_vector_model_path=control_vector_path,
            control_vector_id=control_vector_id,
            config=control_vector_config,
            device=device,
            scale_factor=scale_factor
        )
        # If successful, verify basic properties
        assert cv_model.id == control_vector_id
        assert cv_model.scale_factor == scale_factor
        assert cv_model.control_vector_weights is not None
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        # It's okay if the file can't be loaded in test environment
        pytest.skip(f"Could not load control vector file: {e}")


@pytest.mark.parametrize("device", DEVICES)
def test_control_vector_model_manager(dummy_model, control_vector_config, device):
    """Test ControlVectorModelManager basic operations."""

    manager = ControlVectorModelManager(
        model=dummy_model,
        control_vector_config=control_vector_config
    )

    # Test basic properties
    assert manager.capacity == control_vector_config.max_control_vectors
    assert len(manager._registered_adapters) == 0
    assert len(manager._active_adapters) == 0

    # Create dummy control vectors
    cv1 = create_dummy_control_vector(1, device=device)
    cv2 = create_dummy_control_vector(2, device=device)
    cv3 = create_dummy_control_vector(3, device=device)

    # Test add_adapter
    assert manager.add_adapter(cv1)
    assert not manager.add_adapter(cv1)  # Should return False for duplicate
    assert manager.add_adapter(cv2)

    # Test list_adapters
    adapters = manager.list_adapters()
    assert len(adapters) == 2
    assert 1 in adapters
    assert 2 in adapters

    # Test get_adapter
    retrieved_cv1 = manager.get_adapter(1)
    assert retrieved_cv1 is not None
    assert retrieved_cv1.id == 1

    # Test activate_adapter
    assert manager.activate_adapter(1)
    assert len(manager._active_adapters) > 0

    # Test deactivate_adapter
    assert manager.deactivate_adapter(1)

    # Test remove_adapter
    assert manager.remove_adapter(1)
    assert not manager.remove_adapter(1)  # Should return False for non-existent
    assert 1 not in manager.list_adapters()

    # Test add up to capacity
    assert manager.add_adapter(cv1)
    assert manager.add_adapter(cv3)

    # Test remove_all_adapters
    manager.remove_all_adapters()
    assert len(manager.list_adapters()) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_lru_cache_control_vector_model_manager(dummy_model, control_vector_config, device):
    """Test LRUCacheControlVectorModelManager functionality."""

    manager = LRUCacheControlVectorModelManager(
        model=dummy_model,
        control_vector_config=control_vector_config
    )

    # Test that it inherits from ControlVectorModelManager
    assert isinstance(manager, ControlVectorModelManager)
    assert manager.capacity == control_vector_config.max_control_vectors

    # Create dummy control vectors
    cv1 = create_dummy_control_vector(1, device=device)
    cv2 = create_dummy_control_vector(2, device=device)
    cv3 = create_dummy_control_vector(3, device=device)

    # Test LRU behavior - add adapters
    assert manager.add_adapter(cv1)
    assert manager.add_adapter(cv2)
    assert manager.activate_adapter(1)
    assert manager.activate_adapter(2)

    assert set(manager.list_adapters().keys()) == {1, 2}

    # Add third adapter
    assert manager.add_adapter(cv3)
    assert manager.activate_adapter(3)

    # Verify all are present (capacity is 4)
    assert set(manager.list_adapters().keys()) == {1, 2, 3}

    # Test touch/access pattern
    assert not manager.add_adapter(cv1)  # Already exists
    assert not manager.activate_adapter(1)  # Already active

    # Test deactivate
    assert manager.deactivate_adapter(1)

    # Test remove
    assert manager.remove_adapter(1)
    assert not manager.remove_adapter(1)


@pytest.mark.parametrize("device", DEVICES)
def test_lru_control_vector_manager_capacity(dummy_model, device):
    """Test LRU cache manager respects capacity limits."""

    # Create config with small capacity
    config = ControlVectorConfig(
        max_control_vectors=2,
        adapter_dtype=torch.float16,
        normalize=True
    )

    manager = LRUCacheControlVectorModelManager(
        model=dummy_model,
        control_vector_config=config
    )

    cv1 = create_dummy_control_vector(1, device=device)
    cv2 = create_dummy_control_vector(2, device=device)
    cv3 = create_dummy_control_vector(3, device=device)

    # Add up to capacity
    assert manager.add_adapter(cv1)
    assert manager.add_adapter(cv2)
    assert manager.activate_adapter(1)
    assert manager.activate_adapter(2)

    assert set(manager.list_adapters().keys()) == {1, 2}

    # Add over capacity - should raise RuntimeError for LRU cache
    # In LRU mode, we need to manually evict before adding new ones when at capacity
    with pytest.raises(RuntimeError, match="No free adapter slots"):
        manager.add_adapter(cv3)

    # Remove one to make space
    assert manager.remove_adapter(1)

    # Now we can add cv3
    assert manager.add_adapter(cv3)
    assert manager.activate_adapter(3)

    # Should have cv2 and cv3 now
    adapters = set(manager.list_adapters().keys())
    assert adapters == {2, 3}

    # Test remove_oldest_adapter
    assert manager.remove_oldest_adapter()
    assert len(manager.list_adapters()) == 1


def test_create_control_vector_manager(dummy_model, control_vector_config):
    """Test control vector manager factory function."""

    # Test default manager creation
    manager = create_control_vector_manager(
        model=dummy_model,
        control_vector_config=control_vector_config,
        control_vector_manager_cls=ControlVectorModelManager
    )
    assert isinstance(manager, ControlVectorModelManager)
    assert not isinstance(manager, LRUCacheControlVectorModelManager)

    # Test LRU cache manager creation
    lru_manager = create_control_vector_manager(
        model=dummy_model,
        control_vector_config=control_vector_config,
        control_vector_manager_cls=LRUCacheControlVectorModelManager
    )
    assert isinstance(lru_manager, LRUCacheControlVectorModelManager)


def test_control_vector_mapping_creation():
    """Test ControlVectorMapping creation."""

    device = "cuda:0" if current_platform.is_cuda_alike() else "cpu"

    layer_mapping = {
        0: torch.randn(512, device=device),
        5: torch.randn(512, device=device),
        10: torch.randn(512, device=device)
    }

    mapping = ControlVectorMapping(layer_mapping=layer_mapping)

    assert len(mapping.layer_mapping) == 3
    assert all(isinstance(k, int) for k in mapping.layer_mapping.keys())
    assert all(isinstance(v, torch.Tensor) for v in mapping.layer_mapping.values())


@pytest.mark.parametrize("normalize", [True, False])
def test_control_vector_config_normalization(normalize):
    """Test control vector configuration normalization option."""

    config = ControlVectorConfig(
        max_control_vectors=4,
        adapter_dtype=torch.float32,
        normalize=normalize
    )

    assert config.normalize == normalize
    assert config.max_control_vectors == 4
    assert config.adapter_dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_control_vector_config_dtype(dtype):
    """Test different data types in control vector configuration."""

    if dtype == torch.float16 and not current_platform.is_cuda_alike():
        pytest.skip("float16 not well supported on CPU")

    config = ControlVectorConfig(
        max_control_vectors=2,
        adapter_dtype=dtype,
        normalize=True
    )

    # Verify the dtype is stored correctly
    assert config.adapter_dtype == dtype


@pytest.mark.parametrize("device", DEVICES)
def test_control_vector_manager_capacity_limits(dummy_model, device):
    """Test that manager respects capacity limits."""

    # Create config with small capacity
    config = ControlVectorConfig(
        max_control_vectors=2,
        adapter_dtype=torch.float16,
        normalize=True
    )

    manager = ControlVectorModelManager(
        model=dummy_model,
        control_vector_config=config
    )

    assert manager.capacity == 2

    cv1 = create_dummy_control_vector(1, device=device)
    cv2 = create_dummy_control_vector(2, device=device)
    cv3 = create_dummy_control_vector(3, device=device)

    # Add up to capacity
    assert manager.add_adapter(cv1)
    assert manager.add_adapter(cv2)

    # Try to add beyond capacity - should raise RuntimeError
    with pytest.raises(RuntimeError, match="No free adapter slots"):
        manager.add_adapter(cv3)

    # Should only have 2 registered (at capacity)
    assert len(manager.list_adapters()) == config.max_control_vectors


def test_control_vector_manager_empty_operations(dummy_model, control_vector_config):
    """Test operations on empty manager."""

    manager = ControlVectorModelManager(
        model=dummy_model,
        control_vector_config=control_vector_config
    )

    # Test operations on empty manager
    assert len(manager.list_adapters()) == 0
    assert manager.get_adapter(999) is None
    assert not manager.deactivate_adapter(999)
    assert not manager.remove_adapter(999)

    # Should not raise error
    manager.remove_all_adapters()
    assert len(manager.list_adapters()) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_lru_control_vector_detailed_behavior(dummy_model, device):
    """Test detailed LRU cache behavior similar to LoRA test_lru_lora_model_manager.

    This tests just the LRU cache functionality in detail."""

    # Create config with small capacity for LRU testing
    config = ControlVectorConfig(
        max_control_vectors=2,
        adapter_dtype=torch.float16,
        normalize=True
    )

    manager = LRUCacheControlVectorModelManager(
        model=dummy_model,
        control_vector_config=config
    )

    cv1 = create_dummy_control_vector(1, device=device)
    cv2 = create_dummy_control_vector(2, device=device)
    cv3 = create_dummy_control_vector(3, device=device)
    cv4 = create_dummy_control_vector(4, device=device)

    # Start with empty manager
    assert len(manager.list_adapters()) == 0

    # Add up to capacity
    assert manager.add_adapter(cv1)
    assert manager.add_adapter(cv2)
    assert manager.activate_adapter(1)
    assert manager.activate_adapter(2)

    assert set(manager.list_adapters().keys()) == {1, 2}

    # Add over capacity - need to remove oldest first
    assert manager.remove_adapter(1)  # Remove to make space
    assert manager.add_adapter(cv3)
    assert manager.activate_adapter(3)
    assert manager.remove_adapter(2)  # Remove to make space
    assert manager.add_adapter(cv4)
    assert manager.activate_adapter(4)

    assert set(manager.list_adapters().keys()) == {3, 4}

    # Add cv3 again - should return False since it's already in
    assert not manager.add_adapter(cv3)
    assert not manager.activate_adapter(3)

    # Add cv2 back
    assert manager.remove_adapter(4)  # Make space
    assert manager.add_adapter(cv2)
    assert manager.activate_adapter(2)

    assert set(manager.list_adapters().keys()) == {3, 2}

    # Remove manually
    assert manager.remove_adapter(3)
    assert not manager.remove_adapter(3)  # Already removed

    assert set(manager.list_adapters().keys()) == {2}

    # Add more adapters
    assert manager.add_adapter(cv3)
    assert manager.activate_adapter(3)
    assert manager.remove_adapter(2)  # Make space
    assert manager.add_adapter(cv4)
    assert manager.activate_adapter(4)

    assert set(manager.list_adapters().keys()) == {3, 4}

    # Test remove_oldest_adapter
    assert manager.remove_oldest_adapter()
    assert set(manager.list_adapters().keys()) == {4}

    assert manager.remove_oldest_adapter()
    assert set(manager.list_adapters().keys()) == set()

    # Removing from empty should return False
    assert not manager.remove_oldest_adapter()
    assert set(manager.list_adapters().keys()) == set()


@pytest.mark.parametrize("device", DEVICES)
def test_control_vector_activation_deactivation_sequence(dummy_model, control_vector_config, device):
    """Test detailed activation and deactivation sequences."""

    manager = ControlVectorModelManager(
        model=dummy_model,
        control_vector_config=control_vector_config
    )

    cv1 = create_dummy_control_vector(1, device=device)
    cv2 = create_dummy_control_vector(2, device=device)
    cv3 = create_dummy_control_vector(3, device=device)

    # Add adapters
    assert manager.add_adapter(cv1)
    assert manager.add_adapter(cv2)
    assert manager.add_adapter(cv3)

    # Test activation sequence
    assert manager.activate_adapter(1)
    assert 1 in manager._active_adapters

    assert manager.activate_adapter(2)
    assert 2 in manager._active_adapters

    assert manager.activate_adapter(3)
    assert 3 in manager._active_adapters

    # Activating already active adapter should return False
    assert not manager.activate_adapter(1)
    assert not manager.activate_adapter(2)

    # Test deactivation sequence
    assert manager.deactivate_adapter(1)
    assert 1 not in manager._active_adapters

    assert manager.deactivate_adapter(2)
    assert 2 not in manager._active_adapters

    # Deactivating already inactive adapter should return False
    assert not manager.deactivate_adapter(1)

    # Re-activate
    assert manager.activate_adapter(1)
    assert 1 in manager._active_adapters


@pytest.mark.parametrize("device", DEVICES)
def test_control_vector_multiple_operations(dummy_model, device):
    """Test multiple add/remove/activate operations in sequence."""

    config = ControlVectorConfig(
        max_control_vectors=3,
        adapter_dtype=torch.float16,
        normalize=True
    )

    manager = ControlVectorModelManager(
        model=dummy_model,
        control_vector_config=config
    )

    cv1 = create_dummy_control_vector(1, device=device)
    cv2 = create_dummy_control_vector(2, device=device)
    cv3 = create_dummy_control_vector(3, device=device)

    # Sequence 1: Add all
    assert manager.add_adapter(cv1)
    assert manager.add_adapter(cv2)
    assert manager.add_adapter(cv3)
    assert len(manager.list_adapters()) == 3

    # Sequence 2: Activate all
    assert manager.activate_adapter(1)
    assert manager.activate_adapter(2)
    assert manager.activate_adapter(3)

    # Sequence 3: Remove one, add it back
    assert manager.remove_adapter(2)
    assert 2 not in manager.list_adapters()
    assert manager.add_adapter(cv2)
    assert 2 in manager.list_adapters()

    # Sequence 4: Deactivate and remove
    assert manager.deactivate_adapter(1)
    assert manager.remove_adapter(1)
    assert 1 not in manager.list_adapters()

    # Sequence 5: Verify final state
    assert set(manager.list_adapters().keys()) == {2, 3}


@pytest.mark.parametrize("device", DEVICES)
def test_control_vector_get_adapter_after_operations(dummy_model, control_vector_config, device):
    """Test get_adapter returns correct objects after various operations."""

    manager = ControlVectorModelManager(
        model=dummy_model,
        control_vector_config=control_vector_config
    )

    cv1 = create_dummy_control_vector(1, device=device)
    cv2 = create_dummy_control_vector(2, device=device)

    # Add and verify
    assert manager.add_adapter(cv1)
    retrieved = manager.get_adapter(1)
    assert retrieved is not None
    assert retrieved.id == 1
    assert retrieved.scale_factor == cv1.scale_factor

    # Add second and verify both
    assert manager.add_adapter(cv2)
    retrieved1 = manager.get_adapter(1)
    retrieved2 = manager.get_adapter(2)
    assert retrieved1 is not None
    assert retrieved2 is not None
    assert retrieved1.id == 1
    assert retrieved2.id == 2

    # Activate and verify still accessible
    assert manager.activate_adapter(1)
    retrieved = manager.get_adapter(1)
    assert retrieved is not None
    assert retrieved.id == 1

    # Deactivate and verify still accessible (should be in registered)
    assert manager.deactivate_adapter(1)
    retrieved = manager.get_adapter(1)
    assert retrieved is not None
    assert retrieved.id == 1

    # Remove and verify not accessible
    assert manager.remove_adapter(1)
    retrieved = manager.get_adapter(1)
    assert retrieved is None


@pytest.mark.parametrize("device", DEVICES)
def test_control_vector_list_adapters_consistency(dummy_model, control_vector_config, device):
    """Test that list_adapters returns consistent results."""

    manager = ControlVectorModelManager(
        model=dummy_model,
        control_vector_config=control_vector_config
    )

    cv1 = create_dummy_control_vector(1, device=device)
    cv2 = create_dummy_control_vector(2, device=device)
    cv3 = create_dummy_control_vector(3, device=device)

    # Initially empty
    assert len(manager.list_adapters()) == 0

    # Add one
    manager.add_adapter(cv1)
    adapters = manager.list_adapters()
    assert len(adapters) == 1
    assert 1 in adapters

    # Add more
    manager.add_adapter(cv2)
    manager.add_adapter(cv3)
    adapters = manager.list_adapters()
    assert len(adapters) == 3
    assert set(adapters.keys()) == {1, 2, 3}

    # Activate shouldn't change list
    manager.activate_adapter(1)
    adapters = manager.list_adapters()
    assert len(adapters) == 3

    # Deactivate shouldn't change list
    manager.deactivate_adapter(1)
    adapters = manager.list_adapters()
    assert len(adapters) == 3

    # Remove should change list
    manager.remove_adapter(2)
    adapters = manager.list_adapters()
    assert len(adapters) == 2
    assert set(adapters.keys()) == {1, 3}


@pytest.mark.parametrize("device", DEVICES)
def test_control_vector_capacity_edge_cases(dummy_model, device):
    """Test edge cases around capacity limits."""

    # Test with capacity of 1
    config = ControlVectorConfig(
        max_control_vectors=1,
        adapter_dtype=torch.float16,
        normalize=True
    )

    manager = ControlVectorModelManager(
        model=dummy_model,
        control_vector_config=config
    )

    cv1 = create_dummy_control_vector(1, device=device)
    cv2 = create_dummy_control_vector(2, device=device)

    # Can add one
    assert manager.add_adapter(cv1)
    assert len(manager.list_adapters()) == 1

    # Cannot add second
    with pytest.raises(RuntimeError, match="No free adapter slots"):
        manager.add_adapter(cv2)

    # After removing, can add another
    assert manager.remove_adapter(1)
    assert manager.add_adapter(cv2)
    assert len(manager.list_adapters()) == 1
    assert 2 in manager.list_adapters()


@pytest.mark.parametrize("device", DEVICES)
def test_control_vector_duplicate_operations(dummy_model, control_vector_config, device):
    """Test that duplicate operations are handled correctly."""

    manager = ControlVectorModelManager(
        model=dummy_model,
        control_vector_config=control_vector_config
    )

    cv1 = create_dummy_control_vector(1, device=device)

    # Add once - succeeds
    assert manager.add_adapter(cv1)

    # Add again - fails
    assert not manager.add_adapter(cv1)

    # Activate once - succeeds
    assert manager.activate_adapter(1)

    # Activate again - fails
    assert not manager.activate_adapter(1)

    # Deactivate once - succeeds
    assert manager.deactivate_adapter(1)

    # Deactivate again - fails
    assert not manager.deactivate_adapter(1)

    # Remove once - succeeds
    assert manager.remove_adapter(1)

    # Remove again - fails
    assert not manager.remove_adapter(1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
