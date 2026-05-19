"""
Setup Pytheus metrics backend based on configuration.
"""

from pytheus.backends import load_backend
from pytheus.metrics import Counter, Histogram, Gauge


load_backend()


ACTIVE_REQUESTS = Gauge(
    name='active_requests',
    description='Number of active requests',
    required_labels=['path', 'stream', 'model']
)
INPUT_TOKENS = Histogram(
    name='input_tokens',
    description='Number of total input tokens',
    required_labels=['path', 'stream', 'model'],
    buckets=[50, 100, 200, 300, 400, 500, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
)
OUTPUT_TOKENS = Histogram(
    name='output_tokens',
    description='Number of total output tokens',
    required_labels=['path', 'stream', 'model'],
    buckets=[50, 100, 200, 300, 400, 500, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
)
LOAD_DURATION = Histogram(
    name='load_duration',
    description='Duration of model loading in seconds',
    required_labels=['path', 'stream', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 600.0]
)
INFERENCE_DURATION = Histogram(
    name='inference_duration',
    description='Duration of inference in seconds',
    required_labels=['path', 'stream', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 600.0]
)
FIRST_TOKEN_LATENCY = Histogram(
    name='first_token_latency',
    description='Latency for the first token in seconds',
    required_labels=['path',  'stream', 'model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 600.0]
)
