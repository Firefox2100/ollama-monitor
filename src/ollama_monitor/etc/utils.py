import httpx


HOP_BY_HOP_HEADERS = {
    'connection',
    'keep-alive',
    'proxy-authenticate',
    'proxy-authorization',
    'te',
    'trailer',
    'transfer-encoding',
    'upgrade',
}


def filter_headers(headers) -> dict[str, str]:
    """
    Filter the headers to remove the transient ones
    :param headers: The headers from the upstream/downstream services
    :return: A filtered headers dict
    """
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in HOP_BY_HOP_HEADERS
    }
