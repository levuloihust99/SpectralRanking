import time
from functools import wraps

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def timer(_func=None, *, task_name: Optional[str] = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            ret = func(*args, **kwargs)
            logger.info(
                "{} done in {}s".format(
                    task_name or func.__qualname__, time.perf_counter() - t0
                )
            )
            return ret

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)
