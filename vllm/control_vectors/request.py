# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional

from vllm.adapter_commons.request import AdapterRequest


@dataclass
class ControlVectorRequest(AdapterRequest):
    """
    Request for a Prompt adapter.
    """

    control_vector_name: str
    control_vector_id: int
    control_vector_path: str
    scale: float = 1.0
    base_model_name: Optional[str] = None

    def __hash__(self):
        return super().__hash__()

    @property
    def adapter_id(self):
        return self.control_vector_id

    @property
    def name(self):
        return self.control_vector_name

    @property
    def path(self):
        return self.control_vector_path

    @property
    def scale_factor(self):
        return self.scale
