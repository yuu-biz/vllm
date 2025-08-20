# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define ControlVector functionality mixin for model runners.
"""

from contextlib import contextmanager
from typing import Union

import torch.nn as nn

from vllm.config import ControlVectorConfig
from vllm.logger import init_logger
from vllm.control_vectors.request import ControlVectorRequest
from vllm.control_vectors.worker_manager import LRUCacheWorkerControlVectorManager  # noqa: E501
from vllm.v1.worker.gpu_input_batch import InputBatch as GPUInputBatch
from vllm.v1.worker.tpu_input_batch import InputBatch as TPUInputBatch

InputBatch = Union[TPUInputBatch, GPUInputBatch]

logger = init_logger(__name__)

# Defined as a mixin for GPUModelRunner
class ControlVectorModelRunnerMixin:

    def load_control_vector_model(self, model: nn.Module, 
                                  control_vector_config: ControlVectorConfig, 
                                  device: str) -> nn.Module:

        # Add ControlVector Manager to the Model Runner
        self.control_vector_manager = LRUCacheWorkerControlVectorManager(
            device=device,
            control_vector_config=control_vector_config
        )
        return self.control_vector_manager.create_control_vector_manager(model)

    def _set_active_control_vectors(self, 
                                    control_vector_requests
                                    : set[ControlVectorRequest]) -> None:
        if not self.control_vector_manager:
            raise RuntimeError("ControlVector not enabled.")

        self.control_vector_manager.set_active_adapters(control_vector_requests)


    def set_active_control_vectors(self, input_batch: InputBatch) -> None:
        control_vector_requests: set[ControlVectorConfig]
        control_vector_requests = input_batch.make_control_vector_inputs()
        return self._set_active_control_vectors(control_vector_requests)
    

    def add_control_vector(self, 
                           control_vector_request
                           : ControlVectorRequest) -> bool:
        if not self.control_vector_manager:
            raise RuntimeError("ControlVector is not enabled.")
        return self.control_vector_manager.add_adapter(control_vector_request)

    def remove_control_vector(self, control_vector_id: int) -> bool:
        if not self.control_vector_manager:
            raise RuntimeError("ControlVector is not enabled.")
        return self.control_vector_manager.remove_adapter(control_vector_id)
