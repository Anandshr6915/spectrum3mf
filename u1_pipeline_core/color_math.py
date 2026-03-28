from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def clamp_u8(x: float) -> int:
    return max(0, min(255, int(round(x))))


def rgb_to_hex(rgb: Iterable[int]) -> str:
    r, g, b = list(rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.strip()
    if len(v) != 7 or not v.startswith("#"):
        raise ValueError(f"Invalid hex color: {value}")
    return (int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16))


def srgb_channel_to_linear(x: float) -> float:
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def linear_channel_to_srgb(x: float) -> float:
    c = max(0.0, min(1.0, x))
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055


def blend_weighted_srgb(rgb_colors: list[tuple[int, int, int]], weights: list[int]) -> tuple[int, int, int]:
    total = sum(max(0, int(w)) for w in weights)
    if total <= 0:
        return (0, 0, 0)
    lin = np.zeros(3, dtype=float)
    for (r, g, b), w in zip(rgb_colors, weights):
        ww = max(0, int(w))
        if ww == 0:
            continue
        lin[0] += srgb_channel_to_linear(r / 255.0) * ww
        lin[1] += srgb_channel_to_linear(g / 255.0) * ww
        lin[2] += srgb_channel_to_linear(b / 255.0) * ww
    lin /= float(total)
    return (
        clamp_u8(linear_channel_to_srgb(lin[0]) * 255.0),
        clamp_u8(linear_channel_to_srgb(lin[1]) * 255.0),
        clamp_u8(linear_channel_to_srgb(lin[2]) * 255.0),
    )


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = rgb
    rl = srgb_channel_to_linear(r / 255.0)
    gl = srgb_channel_to_linear(g / 255.0)
    bl = srgb_channel_to_linear(b / 255.0)

    x = (rl * 0.4124 + gl * 0.3576 + bl * 0.1805) / 0.95047
    y = (rl * 0.2126 + gl * 0.7152 + bl * 0.0722)
    z = (rl * 0.0193 + gl * 0.1192 + bl * 0.9505) / 1.08883

    def f(v: float) -> float:
        return v ** (1.0 / 3.0) if v > 0.008856 else (7.787 * v + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)
    return (l, a, b_)


def delta_e_76(lab_a: tuple[float, float, float], lab_b: tuple[float, float, float]) -> float:
    return math.sqrt((lab_a[0] - lab_b[0]) ** 2 + (lab_a[1] - lab_b[1]) ** 2 + (lab_a[2] - lab_b[2]) ** 2)
