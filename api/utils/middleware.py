import logging
from datetime import datetime

from sanic import Sanic

logger = logging.getLogger(__name__)


async def before_request_func(request):
    request.ctx.start_time = datetime.now()


async def after_response_func(request, response):
    logger.info("Total processing time: {}".format(datetime.now() - request.ctx.start_time))


def register_middleware(app: Sanic):
    app.register_middleware(before_request_func, "request")
    app.register_middleware(after_response_func, "response")
