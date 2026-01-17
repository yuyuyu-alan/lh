"""Retry and throttling utilities."""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    max_tries: int = 6,
    base_sleep: float = 1.0,
    max_sleep: float = 30.0,
) -> T:
    """Run a callable with retry and exponential backoff."""
    if max_tries < 1:
        raise ValueError("max_tries must be at least 1")
    if base_sleep <= 0 or max_sleep <= 0:
        raise ValueError("sleep values must be positive")

    last_error: Exception | None = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_tries:
                logger.error("Retry failed after %s attempts: %s", attempt, exc)
                break
            sleep_for = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            logger.warning("Attempt %s failed: %s; retrying in %.2fs", attempt, exc, sleep_for)
            time.sleep(sleep_for)

    if last_error is None:
        raise RuntimeError("Retry failed without capturing an error")
    raise last_error


def throttle(
    min_sleep: float = 0.2,
    max_sleep: float = 0.6,
    sleeper: Callable[[float], None] = time.sleep,
) -> float:
    """Sleep for a random interval between min_sleep and max_sleep."""
    if min_sleep < 0 or max_sleep < 0:
        raise ValueError("sleep values must be non-negative")
    if max_sleep < min_sleep:
        raise ValueError("max_sleep must be >= min_sleep")

    sleep_for = random.uniform(min_sleep, max_sleep)
    logger.debug("Throttling for %.2fs", sleep_for)
    sleeper(sleep_for)
    return sleep_for
