import pytest
from httpx import Request as HttpxRequest, Response as HttpxResponse

from ollama_monitor.router import utils


class FakeRequest:
    def __init__(self, method: str, body: bytes):
        self.method = method
        self.headers = {}
        self._body = body

    async def body(self):
        return self._body


class FailingMetric:
    def labels(self, **_):
        raise RuntimeError('metric backend failed')


class RaisingLogger:
    def __getattr__(self, _):
        def log_method(*_, **__):
            raise RuntimeError('logging backend failed')

        return log_method


class SynchronousClient:
    def __init__(self, response: HttpxResponse):
        self.response = response

    async def request(self, **_):
        return self.response


class StreamingClient:
    def __init__(self, response: HttpxResponse):
        self.response = response
        self.sent_content = None

    def build_request(self, method, url, headers, content):
        self.sent_content = content
        return HttpxRequest(method, url, headers=headers, content=content)

    async def send(self, request, stream=False):
        self.response.request = request
        return self.response


@pytest.mark.asyncio
async def test_synchronous_proxy_preserves_ollama_error_when_telemetry_fails(monkeypatch):
    upstream = HttpxResponse(
        400,
        content=b'{"prompt_eval_count": "not-a-number"}',
        headers={'content-type': 'application/json'},
        request=HttpxRequest('POST', 'http://ollama.test/api/embed'),
    )
    monkeypatch.setattr(utils, 'CLIENT', SynchronousClient(upstream))
    monkeypatch.setattr(utils, 'LOGGER', RaisingLogger())
    monkeypatch.setattr(utils, 'INPUT_TOKENS', FailingMetric())

    response = await utils.transparent_proxy_synchronous(
        request=FakeRequest('POST', b'{"model": "llama"}'),
        ollama_path='/api/embed',
        input_token_field='prompt_eval_count',
    )

    assert response.status_code == 400
    assert response.body == b'{"prompt_eval_count": "not-a-number"}'


@pytest.mark.asyncio
async def test_stream_proxy_forwards_malformed_json_to_ollama(monkeypatch):
    upstream = HttpxResponse(
        400,
        content=b'{"error": "invalid json"}',
        headers={'content-type': 'application/json'},
    )
    client = StreamingClient(upstream)
    monkeypatch.setattr(utils, 'CLIENT', client)

    response = await utils.transparent_proxy_stream(
        request=FakeRequest('POST', b'{"model":'),
        ollama_path='/api/generate',
    )
    body = b''.join([chunk async for chunk in response.body_iterator])

    assert client.sent_content == b'{"model":'
    assert response.status_code == 400
    assert body == b'{"error": "invalid json"}'
