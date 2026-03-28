from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConversionPreset:
    name: str
    max_colors: int
    target_faces: int
    label_smoothing: int
    graph_smoothing: int
    min_island_faces: int


PRESETS: dict[str, ConversionPreset] = {
    "balanced": ConversionPreset(
        name="balanced",
        max_colors=16,
        target_faces=60000,
        label_smoothing=2,
        graph_smoothing=4,
        min_island_faces=35,
    ),
    "photo": ConversionPreset(
        name="photo",
        max_colors=24,
        target_faces=100000,
        label_smoothing=2,
        graph_smoothing=6,
        min_island_faces=22,
    ),
    "text-face": ConversionPreset(
        name="text-face",
        max_colors=32,
        target_faces=140000,
        label_smoothing=1,
        graph_smoothing=2,
        min_island_faces=10,
    ),
}


def get_preset(name: str) -> ConversionPreset:
    key = str(name).strip().lower()
    if key not in PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    return PRESETS[key]

