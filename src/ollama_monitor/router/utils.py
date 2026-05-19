import time
import json
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from httpx import HTTPStatusError

from ollama_monitor.etc.consts import CLIENT, LOGGER
from ollama_monitor.etc.metrics import ACTIVE_REQUESTS, INPUT_TOKENS, OUTPUT_TOKENS, LOAD_DURATION, \
    INFERENCE_DURATION, FIRST_TOKEN_LATENCY
from ollama_monitor.etc.utils import filter_headers


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
    if input_token_field is not None:
        input_tokens = response_data.get(input_token_field)
        if input_tokens is not None:
            INPUT_TOKENS.labels(
                model=model,
                stream=str(stream),
                path=ollama_path,
            ).observe(int(input_tokens))

    if output_token_field is not None:
        output_tokens = response_data.get(output_token_field)
        if output_tokens is not None:
            OUTPUT_TOKENS.labels(
                model=model,
                stream=str(stream),
                path=ollama_path,
            ).observe(int(output_tokens))

    if load_duration_field is not None:
        load_duration = response_data.get(load_duration_field)
        if load_duration is not None:
            LOAD_DURATION.labels(
                model=model,
                stream=str(stream),
                path=ollama_path,
            ).observe(float(load_duration) / duration_divider)

    if inference_duration_field is not None:
        inference_duration = response_data.get(inference_duration_field)
        if inference_duration is not None:
            INFERENCE_DURATION.labels(
                model=model,
                stream=str(stream),
                path=ollama_path,
            ).observe(float(inference_duration) / duration_divider)


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
    LOGGER.info('Sending request to Ollama: %s %s', request.method, ollama_path)
    request_body = await request.body()
    LOGGER.debug('Request data: %s', request_body.decode(errors='replace'))

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

    if request.method == 'POST':
        # Only POST request can carry data and request for processing
        try:
            request_data = json.loads(request_body.decode(errors='replace'))
            response_data = ollama_response.json()
            if input_token_field is not None:
                input_tokens = response_data.get(input_token_field)
                if input_tokens is not None:
                    INPUT_TOKENS.labels(
                        model=request_data.get('model', 'n/a'),
                        stream='false',
                        path=ollama_path,
                    ).observe(int(input_tokens))

            if load_duration_field is not None:
                load_duration = response_data.get(load_duration_field)
                if load_duration is not None:
                    LOAD_DURATION.labels(
                        model=request_data.get('model', 'n/a'),
                        stream='false',
                        path=ollama_path,
                    ).observe(float(load_duration) / duration_divider)
        except Exception as e:
            LOGGER.error('Error occurred while parsing response data: %s', str(e))

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
