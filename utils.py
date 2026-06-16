import random


def _apply_jitter(x: int, y: int, *jitter_params) -> tuple:
    """Apply random jitter to (x,y).

    jitter_params may be empty, one value (both axes), or two values (x,y).
    """
    if not jitter_params:
        mx = my = 5
    elif len(jitter_params) == 1:
        mx = my = int(jitter_params[0])
    else:
        mx = int(jitter_params[0]); my = int(jitter_params[1])

    ox = random.randint(-mx, mx)
    oy = random.randint(-my, my)
    return x + ox, y + oy


def _u16fp(value: float) -> int:
    value = max(0.0, min(1.0, value))
    return int(round(value * 0xFFFF))
