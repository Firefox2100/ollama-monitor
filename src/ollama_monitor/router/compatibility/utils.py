import time
import json
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from httpx import HTTPStatusError

from ollama_monitor.etc.consts import CLIENT, LOGGER
from ollama_monitor.etc.metrics import ACTIVE_REQUESTS, INPUT_TOKENS, OUTPUT_TOKENS, FIRST_TOKEN_LATENCY
from ollama_monitor.etc.utils import filter_headers


async def compatible_stream(request: Request,
                            ollama_path: str,
                            input_token_field: str = None,
                            output_token_field: str = None,
                            count_first_token_latency: bool = False,
                            ) -> StreamingResponse | Response:
    LOGGER.info('Sending request to Ollama: %s %s', request.method, ollama_path)
    request_body = await request.body()
    request_data = json.loads(request_body.decode(errors='replace'))

    try:
        LOGGER.debug('Request data: %s', json.dumps(request_data))
    except Exception as e:
        LOGGER.error('Error occurred while decoding request data: %s', str(e))

    stream = request_data.get('stream', True)

    ACTIVE_REQUESTS.labels(
        model=request_data.get('model', 'n/a'),
        stream=str(stream),
        path=ollama_path,
    ).inc()

    if not stream:
        ollama_response = await CLIENT.request(
            method=request.method,
            url=ollama_path,
            headers=filter_headers(request.headers),
            content=request_body,
        )

        try:
            ollama_response.raise_for_status()
        except HTTPStatusError as e:
            LOGGER.error('Ollama responded with error: %s', str(e), stack_info=True)

        LOGGER.debug(
            'Ollama responded with: %s %s',
            ollama_response.status_code,
            ollama_response.content.decode(errors='replace')
        )
        response_data = ollama_response.json()

        if input_token_field is not None:
            input_tokens = response_data.get('usage', {}).get(input_token_field)
            if input_tokens is not None:
                INPUT_TOKENS.labels(
                    model=request_data.get('model', 'n/a'),
                    stream='false',
                    path=ollama_path,
                ).observe(int(input_tokens))
        if output_token_field is not None:
            output_tokens = response_data.get('usage', {}).get(output_token_field)
            if output_tokens is not None:
                OUTPUT_TOKENS.labels(
                    model=request_data.get('model', 'n/a'),
                    stream='false',
                    path=ollama_path,
                )

        return Response(
            content=ollama_response.content,
            status_code=ollama_response.status_code,
            headers=filter_headers(ollama_response.headers),
            media_type=ollama_response.headers.get('content-type'),
        )

    proxied_request = CLIENT.build_request(
        method=request.method,
        url=ollama_path,
        headers=filter_headers(request.headers),
        content=request_body,
    )
    ollama_response = await CLIENT.send(proxied_request, stream=True)

    try:
        ollama_response.raise_for_status()
    except HTTPStatusError as e:
        LOGGER.error('Ollama responded with error: %s', str(e), stack_info=True)

    async def stream_and_log():
        response_chunks: list[bytes] = []
        starting_time = time.monotonic()
        first_token_time = None

        try:
            async for chunk in ollama_response.aiter_bytes():
                if count_first_token_latency and first_token_time is None:
                    first_token_time = time.monotonic()
                response_chunks.append(chunk)
                LOGGER.debug('Received chunk from Ollama: %s', chunk.decode(errors='replace'))
                yield chunk
        finally:
            ACTIVE_REQUESTS.labels(
                model=request_data.get('model', 'n/a'),
                stream=str(stream),
                path=ollama_path,
            ).dec()

            full_response = b''.join(response_chunks).decode(errors='replace')
            response_lines = [l for l in full_response.splitlines() if l.startswith('data: ')]
            last_message = json.loads(response_lines[-2].split('data: ')[-1])

            if input_token_field is not None:
                input_tokens = last_message.get('usage', {}).get(input_token_field, '')
                if input_tokens is not None:
                    INPUT_TOKENS.labels(
                        model=request_data.get('model', 'n/a'),
                        stream=str(stream),
                        path=ollama_path,
                    ).observe(int(input_tokens))

            if output_token_field is not None:
                output_tokens = last_message.get('usage', {}).get(output_token_field, '')
                if output_tokens is not None:
                    OUTPUT_TOKENS.labels(
                        model=request_data.get('model', 'n/a'),
                        stream=str(stream),
                        path=ollama_path,
                    ).observe(int(output_tokens))

            if count_first_token_latency and first_token_time is not None:
                latency = first_token_time - starting_time
                FIRST_TOKEN_LATENCY.labels(
                    model=request_data.get('model', 'n/a'),
                    stream=str(stream),
                    path=ollama_path,
                ).observe(latency)

            await ollama_response.aclose()

    return StreamingResponse(
        stream_and_log(),
        status_code=ollama_response.status_code,
        headers=filter_headers(ollama_response.headers),
        media_type=ollama_response.headers.get('content-type'),
    )
