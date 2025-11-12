# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from vllm.entrypoints.openai.api_server import models
from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    LoadControlVectorRequest,
    UnloadControlVectorRequest,
)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger

logger = init_logger(__name__)


def register_dynamic_control_vector_routes(router: APIRouter):
    @sagemaker_standards.register_load_adapter_handler(
        request_shape={
            "control_vector_name": "body.name",
            "control_vector_path": "body.src",
        },
    )
    @router.post("/v1/load_control_vector")
    async def load_control_vector(request: LoadControlVectorRequest, raw_request: Request):
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.load_control_vector(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(), status_code=response.error.code)

        return Response(status_code=200, content=response)

    @sagemaker_standards.register_unload_adapter_handler(
        request_shape={
            "control_vector_name": "path_params.adapter_name",
        }
    )
    @router.post("/v1/unload_control_vector")
    async def unload_control_vector(
        request: UnloadControlVectorRequest, raw_request: Request
    ):
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.unload_control_vector(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(), status_code=response.error.code
            )

        return Response(status_code=200, content=response)

    return router
