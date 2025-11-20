from loguru import logger
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime

logger.remove()  # Remove default logger

logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "{extra[request_id]} | " \
           "{message}",
           colorize=True,
           enqueue=True
)

logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[request_id]} | {message}",
    enqueue=True,
)

# Set default request_id for root context
logger = logger.bind(request_id="-")

# Utility: assign request ID per request
def assign_request_id():
    req_id = uuid.uuid4().hex[:8]
    return logger.bind(request_id=req_id)

@contextmanager
def step(description: str):
    start_time = datetime.now()
    step_id = uuid.uuid4().hex[:6]
    
    logger.info(f"[{step_id}] START: {description}")
    try:
        yield
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{step_id}] END: {description} (Duration: {duration:.2f}s)")
    except Exception as e:
        logger.error(f"[{step_id}] ERROR in {description}: {e}")
        raise e