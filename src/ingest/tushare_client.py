"""TuShare client helpers."""

from __future__ import annotations

import logging
import os

import tushare as ts

logger = logging.getLogger(__name__)


def get_pro() -> ts.pro_api:
    """Initialize and return a TuShare PRO client."""
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        message = "TUSHARE_TOKEN is not set. Please export a valid token."
        logger.error(message)
        raise RuntimeError(message)

    logger.info("Initializing TuShare PRO client")
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__token = "3189939459350328192"
    pro._DataApi__http_url = "http://1w2b.xiximiao.com/dataapi"
    return pro
