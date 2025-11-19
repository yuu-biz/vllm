# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch
from torch import nn

from vllm.platforms import current_platform


@dataclass
class ControlVectorMapping:
    layer_mapping: dict[int, torch.Tensor]


class BaseLayerWithControlVector(nn.Module):
    pass


class MLPWithControlVector(BaseLayerWithControlVector):
    def __init__(self, base_layer, hidden_size, dtype) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.normalize = True
        self.control_vectors: dict[int, torch.Tensor | int] = {}
        self.hidden_size = hidden_size
        self.dtype = dtype

        self.active_vector = torch.zeros(
            self.hidden_size, dtype=self.dtype, device=current_platform.device_type
        )

    def set_normalization(self, normalize: bool) -> None:
        self.normalize = normalize

    def set_layer_id(self, layer_id: int) -> None:
        """assign the layer id of this MLP layer"""
        self.layer_id = layer_id

    def set_control_vector(self, index: int, control_vector: torch.Tensor):
        """Set a control vector at a specific index."""
        self.control_vectors[index] = control_vector

    def get_control_vector(self, index: int) -> torch.Tensor | None:
        """Get a control vector by index."""
        return self.control_vectors.get(index)

    def reset_control_vector(self, index: int):
        """Reset a control vector to zero at a specific index."""
        if index in self.control_vectors:
            self.control_vectors[index].copy_(
                torch.zeros(
                    self.hidden_size,
                    dtype=self.dtype,
                    device=current_platform.device_type,
                )
            )

    def set_active_tensor(self, index: int):
        """Sets the active vector"""
        if index is not None and index in self.control_vectors:
            self.active_vector.copy_(self.control_vectors[index])
        else:
            self.active_vector.copy_(
                torch.zeros(
                    self.hidden_size,
                    dtype=self.dtype,
                    device=current_platform.device_type,
                )
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional application of control vectors."""
        hidden_states = self.base_layer(hidden_states)

        norm_pre = torch.norm(hidden_states, dim=-1, keepdim=True)

        hidden_states += self.active_vector

        if self.normalize:
            norm_post = torch.norm(hidden_states, dim=-1, keepdim=True)
            hidden_states = hidden_states * norm_pre / norm_post

        return hidden_states
