import os

import httpx
import pytest
import pytest_asyncio

from ollama_monitor.app import create_app
from ollama_monitor.router import utils as proxy_utils


@pytest.fixture(scope='module')
def ollama_url():
    url = os.getenv('OM_OLLAMA_URL')
    if not url:
        pytest.skip('Set OM_OLLAMA_URL to run live Ollama integration tests')
    return url.rstrip('/')


@pytest_asyncio.fixture
async def live_proxy_client(monkeypatch, ollama_url):
    upstream_client = httpx.AsyncClient(base_url=ollama_url, timeout=120)
    monkeypatch.setattr(proxy_utils, 'CLIENT', upstream_client)

    app = create_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url='http://ollama-monitor.test',
        timeout=120,
    ) as client:
        yield client

    await upstream_client.aclose()


@pytest_asyncio.fixture
async def live_ollama_client(ollama_url):
    async with httpx.AsyncClient(base_url=ollama_url, timeout=120) as client:
        yield client


def _select_model(tags_response: dict) -> str | None:
    configured_model = os.getenv('OM_TEST_MODEL')
    if configured_model:
        return configured_model

    models = [
        model
        for model in tags_response.get('models', [])
        if _looks_like_generation_model(model)
    ]
    if not models:
        return None

    smallest_model = min(models, key=lambda model: model.get('size', 0))
    return smallest_model.get('name')


def _looks_like_generation_model(model: dict) -> bool:
    name = model.get('name', '').lower()
    details = model.get('details', {})
    families = details.get('families') or []
    family = details.get('family')
    model_families = {family, *families}

    if 'bert' in model_families:
        return False

    return not any(
        marker in name
        for marker in ('bge', 'embed', 'embedding', 'minilm')
    )


@pytest.mark.asyncio
async def test_live_proxy_forwards_version(live_proxy_client):
    response = await live_proxy_client.get('/api/version')

    assert response.status_code == 200
    assert response.json()['version']


@pytest.mark.asyncio
async def test_live_proxy_returns_fastapi_404_for_unknown_route(live_proxy_client):
    response = await live_proxy_client.get('/not-an-ollama-route')

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_live_proxy_passes_ollama_error_response_as_is(
        live_proxy_client,
        live_ollama_client,
):
    payload = {'model': '__ollama_monitor_missing_model__'}

    direct_response = await live_ollama_client.post('/api/show', json=payload)
    proxy_response = await live_proxy_client.post('/api/show', json=payload)

    assert proxy_response.status_code == direct_response.status_code
    assert proxy_response.content == direct_response.content


@pytest.mark.asyncio
async def test_live_proxy_generate_with_smallest_available_model(live_proxy_client):
    tags_response = await live_proxy_client.get('/api/tags')
    tags_response.raise_for_status()

    model = _select_model(tags_response.json())
    if model is None:
        pytest.skip('No local Ollama models are available for inference testing')

    response = await live_proxy_client.post(
        '/api/generate',
        json={
            'model': model,
            'prompt': 'Reply with one word: ok',
            'stream': False,
            'options': {
                'num_predict': 1,
            },
        },
    )

    assert response.status_code == 200
    assert response.json()['model'] == model
    assert 'response' in response.json()
