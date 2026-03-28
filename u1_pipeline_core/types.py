from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SlotProfile:
    slot_id: int
    hex: str
    material: str = "PLA"
    brand: str = ""
    label: str = ""
    td: float | None = None


@dataclass(frozen=True)
class MixPolicy:
    max_virtual_slots: int = 40
    max_pattern_len: int = 6
    prefer_2_color_mix: bool = True
    allow_3_4_color_pattern: bool = False


@dataclass(frozen=True)
class FilamentProfile:
    name: str
    version: int
    printer: str
    slots: tuple[SlotProfile, SlotProfile, SlotProfile, SlotProfile]
    mix_policy: MixPolicy


@dataclass
class MeshAsset:
    source_path: Path
    original_input_path: Path
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None
    texture_rgb: np.ndarray | None


@dataclass
class GeometryReport:
    coherent: bool
    components: int
    triangle_count: int
    vertex_count: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    bbox_size: tuple[float, float, float]
    fits_u1_volume: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class PaletteExtraction:
    triangle_rgb: np.ndarray
    palette_rgb: np.ndarray
    triangle_palette_index: np.ndarray


@dataclass
class VirtualMixDefinition:
    filament_id: int
    display_hex: str
    components: tuple[int, ...]
    weights_percent: tuple[int, ...]
    pattern: str


@dataclass
class MappingResult:
    palette_to_filament_id: list[int]
    palette_to_display_hex: list[str]
    virtual_mixes: list[VirtualMixDefinition]


@dataclass
class CompatReport:
    ok: bool
    warnings: list[str]


@dataclass
class ConvertResult:
    output_path: Path
    geometry_report: GeometryReport
    compat_report: CompatReport
    mapping_result: MappingResult
    metadata: dict[str, Any]
