import logging
import time


class FixedPrefixFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, prefix_len: int = 60):
        super().__init__(fmt=fmt or '%(prefix)s %(message)s', datefmt=datefmt)
        self.prefix_len = prefix_len

    def format(self, record: logging.LogRecord) -> str:
        asctime = self.formatTime(record, self.datefmt)
        if asctime and '.' in asctime:
            parts = asctime.rsplit('.', 1)
            if len(parts) == 2:
                main, frac = parts
                asctime = f"{main}.{frac[:3]}"
            else:
                # fallback: truncate to 3 decimals after first dot
                first, *rest = asctime.split('.')
                asctime = f"{first}.{''.join(rest)[:3]}"

        level = record.levelname
        name = record.name
        record.prefix = f"[{asctime}][{level:5s}][{name:5s}]"
        return super().format(record)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Return the creation time of the specified LogRecord as formatted string.

        This implementation always appends milliseconds and avoids using
        unsupported `%f` in time.strftime.
        """
        ct = self.converter(record.created)
        if datefmt:
            try:
                s = time.strftime(datefmt, ct)
            except Exception:
                # fall back: remove any %f and format, we'll append ms below
                safe = datefmt.replace('%f', '')
                s = time.strftime(safe, ct)
        else:
            s = time.strftime('%Y-%m-%d %H:%M:%S', ct)

        # remove any trailing dots introduced by a stripped %f placeholder
        s = s.rstrip('.')

        # append milliseconds
        ms = int(record.msecs) if hasattr(record, 'msecs') else int((record.created - int(record.created)) * 1000)
        return f"{s}.{ms:03d}"


def configure_logging(level=logging.DEBUG, prefix_len: int = 40):
    handler = logging.StreamHandler()
    fmt = '%(prefix)s %(message)s'
    # use datefmt without %f; FixedPrefixFormatter will append milliseconds
    formatter = FixedPrefixFormatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S', prefix_len=prefix_len)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []
    root.addHandler(handler)


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Return a logger configured with the global handlers. Optionally set per-logger level."""
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


__all__ = ["configure_logging", "FixedPrefixFormatter"]
