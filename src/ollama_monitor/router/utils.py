import time
import json
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from httpx import HTTPStatusError

from ollama_monitor.etc.consts import CLIENT, LOGGER
from ollama_monitor.etc.metrics import ACTIVE_REQUESTS, INPUT_TOKENS, OUTPUT_TOKENS, LOAD_DURATION, \
    INFERENCE_DURATION, FIRST_TOKEN_LATENCY
from ollama_monitor.etc.utils import filter_headers


def safe_log(level: str, message: str, *args, **kwargs):
    try:
        getattr(LOGGER, level)(message, *args, **kwargs)
    except Exception:
        pass


def safe_json_loads(data: bytes) -> dict:
    try:
        decoded_data = data.decode(errors='replace')
        parsed_data = json.loads(decoded_data)
        if isinstance(parsed_data, dict):
            return parsed_data
    except Exception as e:
        safe_log('warning', 'Error occurred while decoding request data: %s', str(e))
    return {}


def safe_observe(metric, value, model: str, stream: str | bool, ollama_path: str):
    try:
        metric.labels(
            model=model,
            stream=str(stream),
            path=ollama_path,
        ).observe(value)
    except Exception as e:
        safe_log('warning', 'Error occurred while recording metric: %s', str(e))


def safe_active_requests_delta(model: str, stream: str | bool, ollama_path: str, delta: int):
    try:
        active_requests = ACTIVE_REQUESTS.labels(
            model=model,
            stream=str(stream),
            path=ollama_path,
        )
        if delta > 0:
            active_requests.inc()
        else:
            active_requests.dec()
    except Exception as e:
        safe_log('warning', 'Error occurred while recording active request metric: %s', str(e))


async def safe_close_response(response):
    try:
        await response.aclose()
    except Exception as e:
        safe_log('warning', 'Error occurred while closing Ollama response: %s', str(e))


def monitor_response(response_data: dict,
                     model: str,
                     stream: bool,
                     ollama_path: str,
                     input_token_field: str = None,
                     output_token_field: str = None,
                     load_duration_field: str = None,
                     inference_duration_field: str = None,
                     duration_divider: int = 1_000_000_000,  # nanoseconds
                     ):
    """
    Manipulate the metrics based on the response data
    :param response_data: The response data from the Ollama server as a dictionary
    :param model: The model used for the request
    :param stream: Whether the request is a stream
    :param ollama_path: The path to the ollama server
    :param input_token_field: The field in the response that contains the number of input tokens
    :param output_token_field: The field in the response that contains the number of output tokens
    :param load_duration_field: The field in the response that contains the load duration
    :param inference_duration_field: The field in the response that contains the inference duration
    :param duration_divider: The divider to convert the load duration to seconds
    """
    try:
        if input_token_field is not None:
            input_tokens = response_data.get(input_token_field)
            if input_tokens is not None:
                safe_observe(INPUT_TOKENS, int(input_tokens), model, stream, ollama_path)

        if output_token_field is not None:
            output_tokens = response_data.get(output_token_field)
            if output_tokens is not None:
                safe_observe(OUTPUT_TOKENS, int(output_tokens), model, stream, ollama_path)

        if load_duration_field is not None:
            load_duration = response_data.get(load_duration_field)
            if load_duration is not None:
                safe_observe(
                    LOAD_DURATION,
                    float(load_duration) / duration_divider,
                    model,
                    stream,
                    ollama_path,
                )

        if inference_duration_field is not None:
            inference_duration = response_data.get(inference_duration_field)
            if inference_duration is not None:
                safe_observe(
                    INFERENCE_DURATION,
                    float(inference_duration) / duration_divider,
                    model,
                    stream,
                    ollama_path,
                )
    except Exception as e:
        safe_log('warning', 'Error occurred while parsing response metrics: %s', str(e))


async def transparent_proxy_synchronous(request: Request,
                                        ollama_path: str,
                                        input_token_field: str = None,
                                        load_duration_field: str = None,
                                        duration_divider: int = 1_000_000_000,   # nanoseconds
                                        ) -> Response:
    """
    Synchronously proxy the request to the ollama server

    This method should only be used with Ollama endpoints that returns a single response,
    not a stream, and has text input/output.
    :param request: The request object from FastAPI
    :param ollama_path: The path to the ollama server
    :param input_token_field: The field in the response that contains the number of input tokens
    :param load_duration_field: The field in the response that contains the load duration
    :param duration_divider: The divider to convert the load duration to seconds
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
                input_tokens = response_data.get(input_token_field)
                if input_tokens is not None:
                    safe_observe(
                        INPUT_TOKENS,
                        int(input_tokens),
                        request_data.get('model', 'n/a'),
                        'false',
                        ollama_path,
                    )

            if load_duration_field is not None:
                load_duration = response_data.get(load_duration_field)
                if load_duration is not None:
                    safe_observe(
                        LOAD_DURATION,
                        float(load_duration) / duration_divider,
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


async def transparent_proxy_stream(request: Request,
                                   ollama_path: str,
                                   input_token_field: str = None,
                                   output_token_field: str = None,
                                   load_duration_field: str = None,
                                   inference_duration_field: str = None,
                                   count_first_token_latency: bool = False,
                                   duration_divider: int = 1_000_000_000,   # nanoseconds
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
                monitor_response(
                    response_data=response_data,
                    model=request_data.get('model', 'n/a'),
                    stream=stream,
                    ollama_path=ollama_path,
                    input_token_field=input_token_field,
                    output_token_field=output_token_field,
                    load_duration_field=load_duration_field,
                    inference_duration_field=inference_duration_field,
                    duration_divider=duration_divider,
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
                last_message = json.loads(full_response.splitlines()[-1])

                monitor_response(
                    response_data=last_message,
                    model=request_data.get('model', 'n/a'),
                    stream=stream,
                    ollama_path=ollama_path,
                    input_token_field=input_token_field,
                    output_token_field=output_token_field,
                    load_duration_field=load_duration_field,
                    inference_duration_field=inference_duration_field,
                    duration_divider=duration_divider,
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
