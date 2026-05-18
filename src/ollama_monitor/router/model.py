from fastapi import APIRouter, Request

from .utils import transparent_proxy_synchronous


model_router = APIRouter(
    tags=['Model'],
)


@model_router.post('/api/pull')
async def ollama_api_pull(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/pull',
    )


@model_router.post('/api/push')
async def ollama_api_push(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/push',
    )


@model_router.get('/api/tags')
@model_router.head('/api/tags')
async def ollama_api_tags(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/tags',
    )


@model_router.post('/api/show')
async def ollama_api_show(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/show',
    )


@model_router.delete('/api/delete')
async def ollama_api_delete(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/delete',
    )


@model_router.post('/api/me')
async def ollama_api_me(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/me',
    )


@model_router.post('/api/signout')
async def ollama_api_signout(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/signout',
    )


@model_router.delete('/api/user/keys/{encoded_key}', deprecated=True)
async def ollama_api_delete_key(request: Request,
                                encoded_key: str,
                                ):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path=f'/api/user/keys/{encoded_key}',
    )


@model_router.post('/api/create')
async def ollama_api_create(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/create',
    )


@model_router.post('/api/blobs/{digest}')
@model_router.head('/api/blobs/{digest}')
async def ollama_api_blobs(request: Request,
                           digest: str,
                           ):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path=f'/api/blobs/{digest}',
    )


@model_router.post('/api/copy')
async def ollama_api_copy(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/copy',
    )
