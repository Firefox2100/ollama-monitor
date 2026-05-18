from fastapi import Request, Response
from httpx import HTTPStatusError

from ollama_monitor.etc.consts import CLIENT, LOGGER
from ollama_monitor.etc.utils import filter_headers


async def transparent_proxy_synchronous(request: Request,
                                        ollama_path: str,
                                        ) -> Response:
    """
    Synchronously proxy the request to the ollama server

    This method should only be used with Ollama endpoints that returns a single response,
    not a stream, and has text input/output.
    :param request: The request object from FastAPI
    :param ollama_path: The path to the ollama server
    :return: FastAPI response
    """
    LOGGER.info('Sending request to Ollama: %s %s', request.method, ollama_path)
    request_data = await request.body()
    LOGGER.debug('Request data: %s', request_data.decode(errors='replace'))

    ollama_response = await CLIENT.request(
        method=request.method,
        url=ollama_path,
        headers=filter_headers(request.headers),
        content=request_data,
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
