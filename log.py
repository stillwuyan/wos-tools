import logging
import time


class FixedPrefixFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt=fmt or "%(prefix)s %(message)s", datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        asctime = self.formatTime(record, self.datefmt)
        level = record.levelname
        name = record.name
        record.prefix = f"[{asctime}][{level[:4]:4s}][{name[:6]:6s}]"
        return super().format(record)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        ms = int(record.msecs)
        return f"{s}.{ms:03d}"


def configure_logging(level=logging.INFO):
    handler = logging.StreamHandler()
    formatter = FixedPrefixFormatter(fmt="%(prefix)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(level)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
