from fastapi import APIRouter, Request

from .utils import transparent_proxy_synchronous, transparent_proxy_stream


inference_router = APIRouter(
    tags=['Inference'],
)


@inference_router.get('/api/ps')
async def ollama_api_ps(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/ps',
    )


@inference_router.post('/api/generate')
async def ollama_api_generate(request: Request):
    return await transparent_proxy_stream(
        request=request,
        ollama_path='/api/generate',
        input_token_field='prompt_eval_count',
        output_token_field='eval_count',
        load_duration_field='load_duration',
        inference_duration_field='eval_duration',
        count_first_token_latency=True,
    )


@inference_router.post('/api/chat')
async def ollama_api_chat(request: Request):
    return await transparent_proxy_stream(
        request=request,
        ollama_path='/api/chat',
        input_token_field='prompt_eval_count',
        output_token_field='eval_count',
        load_duration_field='load_duration',
        inference_duration_field='eval_duration',
        count_first_token_latency=True,
    )


@inference_router.post('/api/embed')
async def ollama_api_embed(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/embed',
        input_token_field='prompt_eval_count',
        load_duration_field='load_duration',
    )


@inference_router.post('/api/embeddings', deprecated=True)
async def ollama_api_embeddings(request: Request):
    return await transparent_proxy_synchronous(
        request=request,
        ollama_path='/api/embeddings',
    )
