import logging
import logging.handlers
import os
import sys
from pathlib import Path



ROOT_LOGGER_NAME = "tfm_agent"

CONSOLE_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)-28s │ %(message)s"
CONSOLE_DATE_FORMAT = "%H:%M:%S"

FILE_FORMAT = (
    "%(asctime)s │ %(levelname)-8s │ %(name)-28s │ "
    "%(filename)s:%(lineno)d │ %(message)s"
)
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_DIR = Path("logs")
LOG_FILENAME = LOG_DIR / "agent.log"


_DEFAULT_CONSOLE_LEVEL = "INFO"



_COLORS = {
    "DEBUG":    "\033[36m",
    "INFO":     "\033[32m",
    "WARNING":  "\033[33m",
    "ERROR":    "\033[31m",
    "CRITICAL": "\033[35m",
    "RESET":    "\033[0m",
}


class _ColorFormatter(logging.Formatter):

    def __init__(self, fmt: str, datefmt: str, use_color: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self._use_color:
            color = _COLORS.get(record.levelname, "")
            reset = _COLORS["RESET"]
            record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logging(
    console_level: str | None = None,
    log_file: Path | str | None = None,
    force: bool = False,
) -> logging.Logger:
    root = logging.getLogger(ROOT_LOGGER_NAME)

    if root.handlers and not force:
        return root

    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    _level_str = console_level or os.environ.get("LOG_LEVEL", _DEFAULT_CONSOLE_LEVEL)
    _console_level = getattr(logging, _level_str.upper(), logging.INFO)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(_console_level)

    use_color = sys.stderr.isatty()
    console_handler.setFormatter(
        _ColorFormatter(CONSOLE_FORMAT, CONSOLE_DATE_FORMAT, use_color=use_color)
    )
    root.addHandler(console_handler)

    _log_path = Path(log_file) if log_file else LOG_FILENAME

    if log_file is not False:
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(_log_path),
            when="midnight",
            backupCount=7,
            encoding="utf-8",
            delay=True
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT)
        )
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if not root.handlers:
        setup_logging()
    return logging.getLogger(name)


def log_separator(logger: logging.Logger, label: str = "", level: int = logging.INFO) -> None:
    bar = "─" * 60
    if label:
        padding = max(0, 60 - len(label) - 2) // 2
        line = f"{'─' * padding} {label} {'─' * padding}"
    else:
        line = bar
    logger.log(level, line)
