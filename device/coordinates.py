from __future__ import annotations

from typing import Optional


def window_to_frame(
    x: int, y: int, image_size: tuple[int, int], frame_size: tuple[int, int]
) -> Optional[tuple[int, int]]:
    image_w, image_h = image_size
    frame_w, frame_h = frame_size
    if image_w <= 0 or image_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return None
    if not (0 <= x < image_w and 0 <= y < image_h):
        return None
    return (
        min(frame_w - 1, x * frame_w // image_w),
        min(frame_h - 1, y * frame_h // image_h),
    )


def frame_to_window(
    x: int, y: int, image_size: tuple[int, int], frame_size: tuple[int, int]
) -> tuple[int, int]:
    image_w, image_h = image_size
    frame_w, frame_h = frame_size
    return x * image_w // frame_w, y * image_h // frame_h


def frame_to_device(
    x: int, y: int, frame_size: tuple[int, int], device_size: tuple[int, int]
) -> Optional[tuple[int, int]]:
    frame_w, frame_h = frame_size
    device_w, device_h = device_size
    if frame_w <= 0 or frame_h <= 0 or device_w <= 0 or device_h <= 0:
        return None
    if not (0 <= x < frame_w and 0 <= y < frame_h):
        return None
    return (
        min(device_w - 1, x * device_w // frame_w),
        min(device_h - 1, y * device_h // frame_h),
    )
