# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, Optional

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ControlVectorConfig:
    """Configuration for ControlVectors."""

    max_control_vectors: int = 1
    """Maximum number of ControlVectors in a batch."""
    adapter_dtype: Optional[torch.dtype] = torch.float16
    """Data type for ControlVectors."""
    normalize: bool = False
    """Enable normalization for ControlVectors."""

    def __post_init__(self):
        if self.max_control_vectors < 1:
            raise ValueError("max_control_vectors must be >= 1")