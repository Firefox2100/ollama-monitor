import time
import json
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from httpx import HTTPStatusError

from ollama_monitor.etc.consts import CLIENT
from ollama_monitor.etc.metrics import INPUT_TOKENS, OUTPUT_TOKENS, FIRST_TOKEN_LATENCY
from ollama_monitor.etc.utils import filter_headers
from ollama_monitor.router.utils import (
    safe_active_requests_delta,
    safe_close_response,
    safe_json_loads,
    safe_log,
    safe_observe,
)


async def compatible_stream(request: Request,
                            ollama_path: str,
                            input_token_field: str = None,
                            output_token_field: str = None,
                            count_first_token_latency: bool = False,
                            ) -> StreamingResponse | Response:
    safe_log('info', 'Sending request to Ollama: %s %s', request.method, ollama_path)
    request_body = await request.body()
    request_data = safe_json_loads(request_body)
    safe_log('debug', 'Request data: %s', json.dumps(request_data))

    stream = request_data.get('stream', True)

    safe_active_requests_delta(request_data.get('model', 'n/a'), stream, ollama_path, 1)

    if not stream:
        try:
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
            try:
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
                if output_token_field is not None:
                    output_tokens = response_data.get('usage', {}).get(output_token_field)
                    if output_tokens is not None:
                        safe_observe(
                            OUTPUT_TOKENS,
                            int(output_tokens),
                            request_data.get('model', 'n/a'),
                            'false',
                            ollama_path,
                        )
            except Exception as e:
                safe_log('warning', 'Error occurred while parsing response data: %s', str(e))

            return Response(
                content=ollama_response.content,
                status_code=ollama_response.status_code,
                headers=filter_headers(ollama_response.headers),
                media_type=ollama_response.headers.get('content-type'),
            )
        finally:
            safe_active_requests_delta(request_data.get('model', 'n/a'), stream, ollama_path, -1)

    try:
        proxied_request = CLIENT.build_request(
            method=request.method,
            url=ollama_path,
            headers=filter_headers(request.headers),
            content=request_body,
        )
        ollama_response = await CLIENT.send(proxied_request, stream=True)
    except Exception:
        safe_active_requests_delta(request_data.get('model', 'n/a'), stream, ollama_path, -1)
        raise

    try:
        ollama_response.raise_for_status()
    except HTTPStatusError as e:
        safe_log('error', 'Ollama responded with error: %s', str(e), stack_info=True)

    async def stream_and_log():
        response_chunks: list[bytes] = []
        starting_time = time.monotonic()
        first_token_time = None

        try:
            async for chunk in ollama_response.aiter_bytes():
                if count_first_token_latency and first_token_time is None:
                    first_token_time = time.monotonic()
                response_chunks.append(chunk)
                safe_log('debug', 'Received chunk from Ollama: %s', chunk.decode(errors='replace'))
                yield chunk
        finally:
            safe_active_requests_delta(request_data.get('model', 'n/a'), stream, ollama_path, -1)

            try:
                full_response = b''.join(response_chunks).decode(errors='replace')
                response_lines = [l for l in full_response.splitlines() if l.startswith('data: ')]
                response_messages = [
                    json.loads(line.split('data: ')[-1])
                    for line in response_lines
                    if line != 'data: [DONE]'
                ]
                last_message = response_messages[-1]

                if input_token_field is not None:
                    input_tokens = last_message.get('usage', {}).get(input_token_field)
                    if input_tokens is not None:
                        safe_observe(
                            INPUT_TOKENS,
                            int(input_tokens),
                            request_data.get('model', 'n/a'),
                            stream,
                            ollama_path,
                        )

                if output_token_field is not None:
                    output_tokens = last_message.get('usage', {}).get(output_token_field)
                    if output_tokens is not None:
                        safe_observe(
                            OUTPUT_TOKENS,
                            int(output_tokens),
                            request_data.get('model', 'n/a'),
                            stream,
                            ollama_path,
                        )
            except Exception as e:
                safe_log('warning', 'Error occurred while parsing streamed response data: %s', str(e))

            if count_first_token_latency and first_token_time is not None:
                latency = first_token_time - starting_time
                safe_observe(
                    FIRST_TOKEN_LATENCY,
                    latency,
                    request_data.get('model', 'n/a'),
                    stream,
                    ollama_path,
                )

            await safe_close_response(ollama_response)

    return StreamingResponse(
        stream_and_log(),
        status_code=ollama_response.status_code,
        headers=filter_headers(ollama_response.headers),
        media_type=ollama_response.headers.get('content-type'),
    )
