# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel, Field


class LoadControlVectorRequest(BaseModel):
    control_vector_name: str
    control_vector_path: str
    control_vector_scale: float


class UnloadControlVectorRequest(BaseModel):
    control_vector_name: str
    control_vector_int_id: int | None = Field(default=None)
