# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from typing import Any

import torch

from vllm.config.control_vector import ControlVectorConfig
from vllm.control_vectors.layers import ControlVectorMapping
from vllm.control_vectors.models import (
    ControlVectorModel,
    ControlVectorModelManager,
    LRUCacheControlVectorModelManager,
    create_control_vector_manager,
)
from vllm.control_vectors.request import ControlVectorRequest

logger = logging.getLogger(__name__)


class WorkerControlVectorManager:
    """WorkerControlVectorManager that manages
    control vector models on the worker side.

    Every request, the requested control vectors will be
    loaded (unless they are already loaded),
    and every other control vector will be unloaded."""

    _manager_cls: type[ControlVectorModelManager] = ControlVectorModelManager

    def __init__(
        self,
        device: torch.device,
        control_vector_config: ControlVectorConfig,
        max_num_batched_tokens: int,
        control_vector_model_cls: type[ControlVectorModel] = ControlVectorModel,
    ):
        self._adapter_manager: ControlVectorModelManager
        self._control_vector_model_cls = control_vector_model_cls
        self.control_vector_config = control_vector_config
        self.device = device
        self._max_num_batched_tokens = max_num_batched_tokens

    @property
    def is_enabled(self) -> bool:
        return True

    def create_control_vector_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        control_vector_manager = create_control_vector_manager(
            model,
            control_vector_config=self.control_vector_config,
            max_num_batched_tokens=self._max_num_batched_tokens,
            control_vector_manager_cls=self._manager_cls,
        )
        self._adapter_manager = control_vector_manager
        return control_vector_manager.model

    def _load_adapter(
        self, control_vector_request: ControlVectorRequest
    ) -> ControlVectorModel:
        try:
            control_vector = self._control_vector_model_cls.from_local_checkpoint(
                control_vector_request.control_vector_path,
                control_vector_id=control_vector_request.control_vector_id,
                config=self.control_vector_config,
                device=str(self.device),
                scale_factor=control_vector_request.scale_factor,
            )
        except Exception as e:
            raise RuntimeError(
                f"Loading control vector "
                f"{control_vector_request.control_vector_path}"
                f" failed"
            ) from e
        return control_vector

    def add_dummy_control_vector(
        self, control_vector_request: ControlVectorRequest
    ) -> bool:
        return True

    def pin_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.pin_adapter(adapter_id)

    def set_active_adapters(self, requests: set[Any], token_cv_mapping: tuple[int, ...]) -> None:
        self._apply_adapters(requests)
        self._adapter_manager.set_adapter_mapping(ControlVectorMapping(layer_mapping=token_cv_mapping))

    def add_adapter(self, adapter_request: Any) -> bool:
        if adapter_request.adapter_id in self.list_adapters():
            return False
        loaded_adapter = self._load_adapter(adapter_request)
        loaded = self._adapter_manager.add_adapter(loaded_adapter)
        self._adapter_manager.activate_adapter(loaded_adapter.id)
        return loaded

    def _apply_adapters(self, adapter_requests: set[Any]) -> None:
        existing_adapters = self.list_adapters()
        models_map = {
            adapter_request.adapter_id: adapter_request
            for adapter_request in adapter_requests
            if adapter_request
        }
        if len(models_map) > self._adapter_manager.adapter_slots:
            raise RuntimeError(
                f"Number of requested control vectors "
                f"({len(models_map)}) is greater "
                "than the number of GPU control vector slots "
                f"({self._adapter_manager.adapter_slots})."
            )
        new_adapters = set(models_map.keys())
        adapters_to_add = new_adapters - existing_adapters
        adapters_to_remove = existing_adapters - new_adapters
        for adapter_id in adapters_to_remove:
            self.remove_adapter(adapter_id)
        for adapter_id in adapters_to_add:
            self.add_adapter(models_map[adapter_id])

    def remove_adapter(self, adapter_id: int) -> bool:
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> set[int]:
        return set(self._adapter_manager.list_adapters().keys())


class LRUCacheWorkerControlVectorManager(WorkerControlVectorManager):
    """WorkerControlVectorManager that manages
    control vector models on the worker side.

    Uses an LRU Cache. Every request, the requested
    control vectors will be loaded (unless they are already loaded)
    and least recently used control vectors will
    be unloaded if the cache is above capacity."""

    _control_vector_manager_cls: type[LRUCacheControlVectorModelManager] = (
        LRUCacheControlVectorModelManager
    )

    def create_control_vector_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        control_vector_manager = create_control_vector_manager(
            model,
            control_vector_config=self.control_vector_config,
            max_num_batched_tokens=self._max_num_batched_tokens,
            control_vector_manager_cls=self._control_vector_manager_cls,
        )
        self._adapter_manager: LRUCacheControlVectorModelManager = (
            control_vector_manager
        )
        return control_vector_manager.model

    def _apply_adapters(
        self, control_vector_requests: set[ControlVectorRequest]
    ) -> None:
        models_that_exist = self.list_adapters()
        control_vectors_map = {
            control_vector_request.control_vector_id: control_vector_request
            for control_vector_request in control_vector_requests
            if control_vector_request
        }
        if len(control_vectors_map) > self._adapter_manager.adapter_slots:
            raise RuntimeError(
                f"Number of requested control vectors "
                f"({len(control_vectors_map)}) is greater "
                "than the number of GPU control vector slots "
                f"({self._adapter_manager.adapter_slots})."
            )
        new_adapters = set(control_vectors_map.keys())
        adapters_to_add = new_adapters - models_that_exist
        adapters_to_remove = models_that_exist - new_adapters
        for adapter_id in adapters_to_remove:
            self.remove_adapter(adapter_id)
        for adapter_id in adapters_to_add:
            self.add_adapter(control_vectors_map[adapter_id])

    def add_adapter(self, control_vector_request: ControlVectorRequest) -> bool:
        if control_vector_request.control_vector_id not in self.list_adapters():
            # Remove before we load the new control vector to save memory
            if len(self._adapter_manager) + 1 > self._adapter_manager.capacity:
                self._adapter_manager.remove_oldest_adapter()
            control_vector = self._load_adapter(control_vector_request)
            loaded = self._adapter_manager.add_adapter(control_vector)
        else:
            loaded = self._adapter_manager.get_adapter(
                control_vector_request.adapter_id
            )
        self._adapter_manager.activate_adapter(control_vector_request.control_vector_id)
        return loaded
