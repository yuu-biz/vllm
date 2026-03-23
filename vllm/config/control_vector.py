# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Any

import torch
from pydantic import ConfigDict

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


@config(config=ConfigDict(arbitrary_types_allowed=True))
class ControlVectorConfig:
    """Configuration for ControlVectors."""

    max_control_vectors: int = 1
    """Maximum number of ControlVectors in a batch."""
    adapter_dtype: torch.dtype | None = torch.float16
    """Data type for ControlVectors."""
    normalize: bool = False
    """Enable normalization for ControlVectors."""

    def compute_hash(self) -> str:
        factors: list[Any] = []
        factors.append(self.max_control_vectors)
        factors.append(self.adapter_dtype)
        factors.append(self.normalize)

        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if self.max_control_vectors < 1:
            raise ValueError("max_control_vectors must be >= 1")
