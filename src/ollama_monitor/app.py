from contextlib import asynccontextmanager
import semver
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from httpx import HTTPStatusError, ConnectError
from pytheus.middleware import PytheusMiddlewareASGI
from pytheus.exposition import generate_metrics

from ollama_monitor import __version__
from ollama_monitor.etc.consts import MINIMUM_VERIFIED_VERSION, MAXIMUM_VERIFIED_VERSION, CLIENT, LOGGER
from ollama_monitor.router import experimental_router, inference_router, model_router, status_router, \
    anthropic_compatibility_router, openai_compatibility_router, transparent_proxy_synchronous

@asynccontextmanager
async def lifespan(_):
    try:
        version_response = await CLIENT.get('/api/version')
        version_response.raise_for_status()

        ollama_version = version_response.json()['version']
        ollama_semver = semver.Version.parse(ollama_version)

        if ollama_semver < MINIMUM_VERIFIED_VERSION or ollama_semver > MAXIMUM_VERIFIED_VERSION:
            LOGGER.warning(
                'The configured Ollama version (%s) has not been verified to be compatible with this '
                'version of Ollama Monitor (%s). Please ensure you are using a compatible version of '
                'Ollama to avoid potential issues. Verified versions are between %s and %s (inclusive).',
                ollama_version,
                __version__,
                str(MINIMUM_VERIFIED_VERSION),
                str(MAXIMUM_VERIFIED_VERSION),
            )

        yield
    except (HTTPStatusError, ConnectError):
        LOGGER.exception('Unable to reach the configured Ollama instance. Shutting down.')


def create_app():
    app = FastAPI(
        title='Ollama Monitor',
        version=__version__,
        description='Ollama Monitor: a transparent proxy to monitor all ollama requests.',
        contact={
            'name': 'Firefox2100',
            'url': 'https://www.firefox2100.co.uk/',
            'email': 'wangyunze16@gmail.com',
        },
        license_info={
            'name': 'MIT',
            'url': 'https://github.com/Firefox2100/ollama-monitor/blob/main/LICENSE',
        },
        openapi_tags=[
            {
                'name': 'Experimental',
                'description': 'Endpoints that are not yet officially released by Ollama. They are subject '
                               'to change without notice, thus might break with this proxy server too.'
            },
            {
                'name': 'Inference',
                'description': 'Endpoints for interacting with Ollama inference.',
            },
            {
                'name': 'Model',
                'description': 'Endpoints for managing Ollama models.',
            },
            {
                'name': 'Status',
                'description': 'Endpoints for checking the status and version of the Ollama server.',
            },
            {
                'name': 'Anthropic Compatibility',
                'description': 'Endpoints for compatibility with Anthropic API.'
            },
            {
                'name': 'OpenAI Compatibility',
                'description': 'Endpoints for compatibility with OpenAI API.'
            }
        ],
        lifespan=lifespan,
        redirect_slashes=False,
    )

    app.add_middleware(PytheusMiddlewareASGI)

    app.include_router(experimental_router)
    app.include_router(inference_router)
    app.include_router(model_router)
    app.include_router(status_router)
    app.include_router(anthropic_compatibility_router)
    app.include_router(openai_compatibility_router)

    @app.get('', tags=['Status'])
    @app.head('', tags=['Status'])
    @app.get('/', tags=['Status'])
    @app.head('/', tags=['Status'])
    async def ollama_root(request: Request):
        return await transparent_proxy_synchronous(
            request=request,
            ollama_path='',
        )

    @app.get('/metrics', response_class=PlainTextResponse)
    async def metrics():
        return generate_metrics()

    return app


app = create_app()
