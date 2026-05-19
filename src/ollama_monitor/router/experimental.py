from fastapi import APIRouter, Request

from .utils import transparent_proxy_synchronous


experimental_router = APIRouter(
    tags=['Experimental'],
)


@experimental_router.post('/api/experimental/web_search')
async def ollama_api_web_search(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/experimental/web_search',
    )


@experimental_router.post('/api/experimental/web_fetch')
async def ollama_api_web_fetch(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/experimental/web_fetch',
    )


@experimental_router.get('/api/experimental/model-recommendations')
async def ollama_api_model_recommendations(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/experimental/model-recommendations',
    )
