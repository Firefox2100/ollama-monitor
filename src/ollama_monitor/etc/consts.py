"""
Constants and configuration settings for the Biomedical Terminology Service.
"""

import os
import logging
import sys
from copy import copy
from typing import Literal
import semver
import click
from httpx import AsyncClient


_log_level = os.getenv('OM_LOG_LEVEL', 'INFO')
_ollama_url = os.getenv('OM_OLLAMA_URL', 'http://localhost:11434').strip('/')

LOGGER = logging.getLogger('ollama-monitor')
LOGGER.setLevel(_log_level)

if not LOGGER.hasHandlers():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(_log_level)


    class ColourizedFormatter(logging.Formatter):
        """
        This class is copied from the uvicorn library to keep it consistent.
        Credit and copyright belongs to the original authors.
        """

        level_name_colors = {
            logging.DEBUG: lambda level_name: click.style(str(level_name), fg="cyan"),
            logging.INFO: lambda level_name: click.style(str(level_name), fg="green"),
            logging.WARNING: lambda level_name: click.style(str(level_name), fg="yellow"),
            logging.ERROR: lambda level_name: click.style(str(level_name), fg="red"),
            logging.CRITICAL: lambda level_name: click.style(str(level_name), fg="bright_red"),
        }

        def __init__(
                self,
                fmt: str | None = None,
                datefmt: str | None = None,
                style: Literal["%", "{", "$"] = "%",
                use_colors: bool | None = None,
        ):
            if use_colors in (True, False):
                self.use_colors = use_colors
            else:
                self.use_colors = sys.stdout.isatty()
            super().__init__(fmt=fmt, datefmt=datefmt, style=style)

        def color_level_name(self, level_name: str, level_no: int) -> str:
            def default(level_name: str) -> str:
                return str(level_name)  # pragma: no cover

            func = self.level_name_colors.get(level_no, default)
            return func(level_name)

        def should_use_colors(self) -> bool:
            return True  # pragma: no cover

        def formatMessage(self, record: logging.LogRecord) -> str:
            recordcopy = copy(record)
            levelname = recordcopy.levelname
            separator = " " * (8 - len(recordcopy.levelname))
            if self.use_colors:
                levelname = self.color_level_name(levelname, recordcopy.levelno)
                if "color_message" in recordcopy.__dict__:
                    recordcopy.msg = recordcopy.__dict__["color_message"]
                    recordcopy.__dict__["message"] = recordcopy.getMessage()
            recordcopy.__dict__["levelprefix"] = levelname + ":" + separator
            return super().formatMessage(recordcopy)


    class DefaultFormatter(ColourizedFormatter):
        """
        This class is copied from the uvicorn library to keep it consistent.
        Credit and copyright belongs to the original authors.
        """

        def should_use_colors(self) -> bool:
            return sys.stderr.isatty()  # pragma: no cover

    formatter = DefaultFormatter(
        fmt='%(levelprefix)s %(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console_handler.setFormatter(formatter)

    LOGGER.addHandler(console_handler)

CLIENT = AsyncClient(
    base_url=_ollama_url,
    timeout=None,
)

MINIMUM_VERIFIED_VERSION = semver.Version.parse('0.23.3')
MAXIMUM_VERIFIED_VERSION = semver.Version.parse('0.23.3')
