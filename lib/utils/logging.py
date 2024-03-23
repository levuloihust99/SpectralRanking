import logging

RED = "\x1b[38;5;9m"
BLUE = "\x1b[38;5;6m"
YELLOW = "\x1b[38;5;3m"
GRAY = "\x1b[38;5;8m"
BOLD_GREEN = "\x1b[38;5;2;1m"
GREEN = "\x1b[38;5;2m"
RESET_CODE = "\x1b[0m"


class ColorFormatter(logging.Formatter):

    ERROR_FORMATTER = (
        f"{GREEN}%(asctime)s - %(name)s [%(process)s]{RESET_CODE} "
        f"[{RED}%(levelname)s{RESET_CODE}] {RED}%(message)s{RESET_CODE}"
    )
    INFO_FORMATTER = (
        f"{GREEN}%(asctime)s - %(name)s [%(process)s]{RESET_CODE} "
        f"[{BLUE}%(levelname)s{RESET_CODE}] {BLUE}%(message)s{RESET_CODE}"
    )
    WARNING_FORMATTER = (
        f"{GREEN}%(asctime)s - %(name)s [%(process)s]{RESET_CODE} "
        f"[{YELLOW}%(levelname)s{RESET_CODE}] {YELLOW}%(message)s{RESET_CODE}"
    )
    DEBUG_FORMATTER = (
        f"{GREEN}%(asctime)s - %(name)s [%(process)s]{RESET_CODE} "
        f"[{GRAY}%(levelname)s{RESET_CODE}] {GRAY}%(message)s{RESET_CODE}"
    )
    DEFAULT_FORMATTER = (
        f"{GREEN}%(asctime)s - %(name)s [%(process)s] [%(levelname)s] %(message)s{RESET_CODE}"
    )

    def format(self, record):
        if record.levelno == logging.INFO:
            formatter = logging.Formatter(self.INFO_FORMATTER)
        elif record.levelno == logging.ERROR:
            formatter = logging.Formatter(self.ERROR_FORMATTER)
        elif record.levelno == logging.WARNING:
            formatter = logging.Formatter(self.WARNING_FORMATTER)
        elif record.levelno == logging.DEBUG:
            formatter = logging.Formatter(self.DEBUG_FORMATTER)
        else:
            formatter = logging.Formatter(self.DEBUG_FORMATTER)
        return formatter.format(record)


def add_color_formatter(logger: logging.Logger):
    handlers = logger.handlers
    for handler in handlers:
        handler.setFormatter(ColorFormatter())


def do_setup_logging(level=None):
    if level is not None:
        if isinstance(level, str):
            level = get_log_level(level)
        logging.basicConfig(level=level)
    add_color_formatter(logging.root)


def get_log_level(level: str):
    log_level = None
    if level.lower() == "info":
        log_level = logging.INFO
    elif level.lower() == "debug":
        log_level = logging.DEBUG
    elif level.lower() == "warning":
        log_level = logging.WARNING
    elif level.lower() == "error":
        log_level = logging.ERROR
    elif level.lower() == "critical":
        log_level = logging.CRITICAL
    else:
        log_level = logging.NOTSET
    return log_level
