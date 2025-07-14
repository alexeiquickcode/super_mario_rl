import logging
import threading
from logging.handlers import (
    QueueHandler,
    QueueListener,
)
from multiprocessing import Queue
from typing import Any


def setup_log_queue() -> "Queue[Any]":
    log_queue: "Queue[Any]" = Queue()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(world_stage)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    listener = QueueListener(log_queue, console_handler)
    listener.start()

    return log_queue


class SubprocessLogger:
    """Logger wrapper that adds world_stage context to log messages."""

    def __init__(self, logger: logging.Logger, world_stage: str):
        self.logger = logger
        self.world_stage = world_stage

    def info(self, msg: str) -> None:
        self.logger.info(msg, extra={"world_stage": self.world_stage})

    def error(self, msg: str) -> None:
        self.logger.error(msg, extra={"world_stage": self.world_stage})

    def warning(self, msg: str) -> None:
        self.logger.warning(msg, extra={"world_stage": self.world_stage})


def log_subprocess(log_queue: "Queue[Any]", world_stage: str = "X-Y") -> SubprocessLogger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Avoid duplicate handlers
    handler = QueueHandler(log_queue)
    logger.addHandler(handler)
    logger.propagate = False

    return SubprocessLogger(logger, world_stage)


class LoggerManager:
    """
    Centralized logger manager for thread-safe access to level-specific loggers.

    Usage:
        logger = logger_manager.get_logger("1-1")
        logger.info('THIS IS A LOG')
    """

    def __init__(self):
        self.log_queue = setup_log_queue()
        self._loggers: dict[str, SubprocessLogger] = {}
        self._lock = threading.Lock()

    def get_logger(self, world_stage: str = "main") -> SubprocessLogger:
        """Get or create a logger for the specified world stage."""
        with self._lock:
            if world_stage not in self._loggers:
                self._loggers[world_stage] = log_subprocess(self.log_queue, world_stage)
            return self._loggers[world_stage]

    def get_level_logger(self, world: int, stage: int) -> SubprocessLogger:
        """Convenience method to get logger for a specific world-stage."""
        return self.get_logger(f"{world}-{stage}")


# Global instance
logger_manager = LoggerManager()
