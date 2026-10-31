import time


def check_cancelled(event):
    if event is not None and event.is_set():
        raise KeyboardInterrupt


def sleep(seconds, event=None):
    if event is None:
        time.sleep(seconds)
    elif event.wait(seconds):
        raise KeyboardInterrupt
