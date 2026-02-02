# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import APIRouter, Request, FastAPI
from fastapi.responses import JSONResponse, Response

from vllm import envs
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)
from vllm.entrypoints.openai.models.api_router import models
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.control_vector.protocol import (
    LoadControlVectorRequest,
    UnloadControlVectorRequest,
)
from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()


def attach_router(app: FastAPI):
    if not envs.VLLM_ALLOW_RUNTIME_CONTROL_VECTOR_UPDATING:
        """If Control Vector dynamic loading & unloading is not enabled, do nothing."""
        return
    logger.warning(
        "Control Vector dynamic loading & unloading is enabled in the API "
        "server. This should ONLY be used for local development!"
    )

    @sagemaker_standards.register_load_adapter_handler(
        request_shape={
            "control_vector_name": "body.name",
            "control_vector_path": "body.src",
        },
    )

    @router.post("/v1/load_control_vector")
    async def load_control_vector(
        request: LoadControlVectorRequest, raw_request: Request
    ):
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.load_control_vector(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(), status_code=response.error.code
            )

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

    # register the router
    app.include_router(router)