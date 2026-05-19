from fastapi import APIRouter, Request
from .utils import compatible_stream


anthropic_compatibility_router = APIRouter(
    tags=['Anthropic Compatibility'],
)


@anthropic_compatibility_router.post('/v1/messages')
async def anthropic_api_messages(request: Request):
    return await compatible_stream(
        request=request,
        ollama_path='/v1/messages',
        input_token_field='input_tokens',
        output_token_field='output_tokens',
        count_first_token_latency=True,
    )
