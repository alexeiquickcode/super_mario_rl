import os
import sys
import traceback
from typing import TypeVar

from loguru import logger as loguru_logger

LOGLEVEL = os.getenv('LOGLEVEL', 'INFO').upper()
T = TypeVar('T')


def setup_logger():
    """Sets up the loguru logger."""
    loguru_logger.remove(handler_id=None)  # Removes existing all loggers.
    loguru_logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<m><b>JOBS</b></m> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        " - <level>{message}</level>",
        level=LOGLEVEL,
    )
    return


def get_traceback(exc: Exception, ex_traceback=None) -> tuple[str, str]:
    """Returns the traceback message of an exception."""

    if ex_traceback is None:
        ex_traceback = exc.__traceback__
    tb_lines: list[str] = [line.rstrip('\n') for line in traceback.format_exception(exc.__class__, exc, ex_traceback)]
    formatted_tb_lines: list[str] = [line.replace('\n', ' ').replace('\t', ' ') for line in tb_lines]
    message: str = f'{exc}. {" ;; ".join(formatted_tb_lines)}'
    top_level_exception: str = tb_lines[-1]

    return message, top_level_exception
