import sys
import os

try:
    import uvicorn
    import uvicorn.logging
except (ImportError, ModuleNotFoundError):
    print(
        'uvicorn is not installed. Either install it with pip install uvicorn, or manually use another ASGI server.'
    )
    sys.exit(1)

from ollama_monitor.app import create_app


def main():
    app = create_app()
    host = str(os.getenv('HOST', '0.0.0.0'))
    port = int(os.getenv('OM_PORT', 11435))

    uvicorn.run(
        app,
        host=host,
        port=port,
    )


if __name__ == '__main__':
    main()
