from fastapi import APIRouter, Request

from .utils import transparent_proxy_synchronous


status_router = APIRouter(
    tags=['Status'],
)


@status_router.get('/api/status')
async def ollama_api_status(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/status',
    )


@status_router.get('/api/version')
@status_router.head('/api/version')
async def ollama_api_version(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/version',
    )
