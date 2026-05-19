from fastapi import APIRouter, Request
from fastapi.responses import Response
from httpx import HTTPStatusError

from ollama_monitor.etc.consts import CLIENT
from ollama_monitor.etc.metrics import INPUT_TOKENS
from ollama_monitor.etc.utils import filter_headers
from ollama_monitor.router.utils import safe_json_loads, safe_log, safe_observe
from .utils import compatible_stream


openai_compatibility_router = APIRouter(
    tags=['OpenAI Compatibility'],
)


async def openai_compatible_synchronous(request: Request,
                                        ollama_path: str,
                                        input_token_field: str = None,
                                        ) -> Response:
    """
    Synchronously proxy the request to the ollama server's OpenAI compatibility endpoints
    :param request: The request object from FastAPI
    :param ollama_path: The path to the ollama server
    :param input_token_field: The field in the response that contains the number of input tokens
    :return: FastAPI response object
    """
    safe_log('info', 'Sending request to Ollama: %s %s', request.method, ollama_path)
    request_body = await request.body()
    safe_log('debug', 'Request data: %s', request_body.decode(errors='replace'))

    ollama_response = await CLIENT.request(
        method=request.method,
        url=ollama_path,
        headers=filter_headers(request.headers),
        content=request_body,
    )

    try:
        ollama_response.raise_for_status()
    except HTTPStatusError as e:
        safe_log('error', 'Ollama responded with error: %s', str(e), stack_info=True)

    safe_log(
        'debug',
        'Ollama responded with: %s %s',
        ollama_response.status_code,
        ollama_response.content.decode(errors='replace')
    )

    if request.method == 'POST':
        # Only POST request can carry data and request for processing
        try:
            request_data = safe_json_loads(request_body)
            response_data = ollama_response.json()
            if input_token_field is not None:
                input_tokens = response_data.get('usage', {}).get(input_token_field)
                if input_tokens is not None:
                    safe_observe(
                        INPUT_TOKENS,
                        int(input_tokens),
                        request_data.get('model', 'n/a'),
                        'false',
                        ollama_path,
                    )

        except Exception as e:
            safe_log('warning', 'Error occurred while parsing response data: %s', str(e))

    headers = filter_headers(ollama_response.headers)

    if request.method == 'HEAD':
        # Preserve upstream headers/status, but do not send a body.
        return Response(
            status_code=ollama_response.status_code,
            headers=headers,
        )

    return Response(
        content=ollama_response.content,
        status_code=ollama_response.status_code,
        headers=headers,
        media_type=ollama_response.headers.get('content-type'),
    )


@openai_compatibility_router.post('/v1/chat/completions')
async def openai_api_chat_completions(request: Request):
    return await compatible_stream(
        request=request,
        ollama_path='/v1/chat/completions',
        input_token_field='prompt_tokens',
        output_token_field='completion_tokens',
        count_first_token_latency=True,
    )


@openai_compatibility_router.post('/v1/completions')
async def openai_api_completions(request: Request):
    return await compatible_stream(
        request=request,
        ollama_path='/v1/completions',
        input_token_field='prompt_tokens',
        output_token_field='completion_tokens',
        count_first_token_latency=True,
    )


@openai_compatibility_router.post('/v1/embeddings')
async def openai_api_embeddings(request: Request):
    return await openai_compatible_synchronous(
        request=request,
        ollama_path='/v1/embeddings',
        input_token_field='prompt_tokens',
    )


@openai_compatibility_router.get('/v1/models')
async def openai_api_models(request: Request):
    return await openai_compatible_synchronous(
        request=request,
        ollama_path='/v1/models',
    )


@openai_compatibility_router.get('/v1/models/{model_name}')
async def openai_api_model(request: Request,
                           model_name: str,
                           ):
    return await openai_compatible_synchronous(
        request=request,
        ollama_path=f'/v1/models/{model_name}',
    )


@openai_compatibility_router.post('/v1/responses')
async def openai_api_responses(request: Request):
    return await compatible_stream(
        request=request,
        ollama_path='/v1/responses',
        input_token_field='prompt_tokens',
        output_token_field='response_tokens',
        count_first_token_latency=True,
    )


@openai_compatibility_router.post('/v1/images/generations')
async def openai_api_images_generations(request: Request):
    return await openai_compatible_synchronous(
        request=request,
        ollama_path='/v1/images/generations',
    )


@openai_compatibility_router.post('/v1/images/edits')
async def openai_api_images_edits(request: Request):
    return await openai_compatible_synchronous(
        request=request,
        ollama_path='/v1/images/edits',
    )


@openai_compatibility_router.post('/v1/audio/transcriptions')
async def openai_api_audio_transcriptions(request: Request):
    # This endpoint accepts multipart/form-data, and respond with JSON or plaintext
    safe_log('info', 'Sending request to Ollama: POST /v1/audio/transcriptions')
    request_body = await request.body()
    safe_log('debug', 'Request data: %s', request_body.decode(errors='replace'))

    ollama_response = await CLIENT.request(
        method=request.method,
        url='/v1/audio/transcriptions',
        headers=filter_headers(request.headers),
        content=request_body,
    )

    try:
        ollama_response.raise_for_status()
    except HTTPStatusError as e:
        safe_log('error', 'Ollama responded with error: %s', str(e), stack_info=True)

    safe_log(
        'debug',
        'Ollama responded with: %s %s',
        ollama_response.status_code,
        ollama_response.content.decode(errors='replace')
    )

    return Response(
        content=ollama_response.content,
        status_code=ollama_response.status_code,
        headers=filter_headers(ollama_response.headers),
        media_type=ollama_response.headers.get('content-type'),
    )
