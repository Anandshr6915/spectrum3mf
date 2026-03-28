from __future__ import annotations

from pathlib import Path

import numpy as np

from .asset_ingest import load_asset
from .color_extract import extract_palette
from .color_math import delta_e_76, hex_to_rgb, rgb_to_lab
from .compat_v094 import validate_mapping_v094
from .geometry_guard import inspect_geometry
from .label_smooth import (
    optimize_labels_graph_energy,
    remove_small_label_islands,
    smooth_face_labels_edge_aware,
    smooth_face_labels_uv_aware,
    suppress_thin_label_chains,
)
from .mesh_refine import refine_mesh_with_uv
from .mix_optimizer import assign_colors_to_filaments, map_palette_to_filaments
from .profile_manager import FilamentProfile
from .threemf_writer import write_3mf
from .types import ConvertResult

_PAINT_SAFE_MAX_FILAMENT_ID = 18


def _scale_vertices_to_target_longest_edge(vertices: np.ndarray, target_longest_edge_mm: float) -> tuple[np.ndarray, float]:
    if target_longest_edge_mm <= 0:
        return vertices, 1.0
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    size = maxs - mins
    longest = float(np.max(size))
    if longest <= 0:
        return vertices, 1.0
    scale = float(target_longest_edge_mm) / longest
    return vertices * scale, scale


def _paint_safe_remap_ids(
    *,
    triangle_filament_ids: np.ndarray,
    profile: FilamentProfile,
    mapping_display_hex_by_id: dict[int, str],
    max_filament_id: int = _PAINT_SAFE_MAX_FILAMENT_ID,
) -> tuple[np.ndarray, int]:
    if triangle_filament_ids.size == 0:
        return triangle_filament_ids, 0
    used_ids = sorted(set(int(i) for i in triangle_filament_ids.tolist()))
    if all(i <= max_filament_id for i in used_ids):
        return triangle_filament_ids, 0

    allowed_ids = [i for i in used_ids if i <= max_filament_id]
    if not allowed_ids:
        allowed_ids = [1, 2, 3, 4]

    # Ensure physical slots always available.
    for sid in (1, 2, 3, 4):
        if sid not in allowed_ids:
            allowed_ids.append(sid)
    allowed_ids = sorted(set(allowed_ids))

    allowed_lab = {
        fid: rgb_to_lab(hex_to_rgb(mapping_display_hex_by_id.get(fid, "#808080")))
        for fid in allowed_ids
    }
    original_lab = {
        fid: rgb_to_lab(hex_to_rgb(mapping_display_hex_by_id.get(fid, "#808080")))
        for fid in used_ids
    }

    remap: dict[int, int] = {}
    changed = 0
    for fid in used_ids:
        if fid <= max_filament_id:
            remap[fid] = fid
            continue
        best = allowed_ids[0]
        best_de = float("inf")
        for aid in allowed_ids:
            de = delta_e_76(original_lab[fid], allowed_lab[aid])
            if de < best_de:
                best_de = de
                best = aid
        remap[fid] = best
        changed += 1

    out = np.asarray([remap[int(fid)] for fid in triangle_filament_ids], dtype=np.int32)
    return out, changed


def convert_asset_to_3mf(
    *,
    input_path: Path,
    profile: FilamentProfile,
    output_path: Path,
    max_colors: int = 12,
    layer_height_mm: float = 0.08,
    target_longest_edge_mm: float = 80.0,
    target_faces: int = 30000,
    label_smoothing_iterations: int = 3,
    min_island_faces: int = 45,
    graph_smoothing_iterations: int = 4,
    uv_smoothing_iterations: int = 2,
    export_mode: str = "paint",
) -> ConvertResult:
    asset = load_asset(input_path)
    scaled_vertices, scale_factor = _scale_vertices_to_target_longest_edge(
        asset.vertices, target_longest_edge_mm
    )
    asset.vertices = scaled_vertices
    refined_vertices, refined_faces, refined_uv = refine_mesh_with_uv(
        vertices=asset.vertices,
        faces=asset.faces,
        uv=asset.uv,
        target_faces=target_faces,
    )
    asset.vertices = refined_vertices
    asset.faces = refined_faces
    asset.uv = refined_uv

    geom = inspect_geometry(asset)
    palette = extract_palette(asset, max_colors=max_colors)

    palette_list = [tuple(int(c) for c in row) for row in palette.palette_rgb]
    mapping = map_palette_to_filaments(palette_list, profile)
    compat = validate_mapping_v094(mapping, profile)

    triangle_filament_ids = np.asarray(
        assign_colors_to_filaments(
            [tuple(int(c) for c in row) for row in palette.triangle_rgb],
            profile,
            mapping,
        ),
        dtype=np.int32,
    )
    triangle_filament_ids = smooth_face_labels_edge_aware(
        vertices=asset.vertices,
        faces=asset.faces,
        labels=triangle_filament_ids,
        face_rgb=palette.triangle_rgb,
        iterations=label_smoothing_iterations,
        color_edge_delta_e=14.0,
    )
    label_rgb_by_id: dict[int, tuple[int, int, int]] = {
        int(slot.slot_id): hex_to_rgb(slot.hex) for slot in profile.slots
    }
    for vm in mapping.virtual_mixes:
        label_rgb_by_id[int(vm.filament_id)] = hex_to_rgb(vm.display_hex)
    triangle_filament_ids = optimize_labels_graph_energy(
        vertices=asset.vertices,
        faces=asset.faces,
        labels=triangle_filament_ids,
        face_rgb=palette.triangle_rgb,
        label_rgb_by_id=label_rgb_by_id,
        iterations=graph_smoothing_iterations,
        smooth_weight=2.4,
        confidence_margin=9.0,
    )
    triangle_filament_ids = smooth_face_labels_uv_aware(
        faces=asset.faces,
        uv=asset.uv,
        labels=triangle_filament_ids,
        face_rgb=palette.triangle_rgb,
        iterations=uv_smoothing_iterations,
        uv_radius=0.018,
        color_edge_delta_e=11.0,
    )
    triangle_filament_ids = remove_small_label_islands(
        vertices=asset.vertices,
        faces=asset.faces,
        labels=triangle_filament_ids,
        face_rgb=palette.triangle_rgb,
        min_component_faces=min_island_faces,
        iterations=2,
        color_edge_delta_e=16.0,
    )
    triangle_filament_ids = suppress_thin_label_chains(
        vertices=asset.vertices,
        faces=asset.faces,
        labels=triangle_filament_ids,
        min_same_neighbors=2,
        iterations=3,
    )
    paint_safe_remapped_count = 0
    if export_mode == "paint":
        display_hex_by_id: dict[int, str] = {int(s.slot_id): s.hex.upper() for s in profile.slots}
        for vm in mapping.virtual_mixes:
            display_hex_by_id[int(vm.filament_id)] = vm.display_hex
        triangle_filament_ids, paint_safe_remapped_count = _paint_safe_remap_ids(
            triangle_filament_ids=triangle_filament_ids,
            profile=profile,
            mapping_display_hex_by_id=display_hex_by_id,
            max_filament_id=_PAINT_SAFE_MAX_FILAMENT_ID,
        )

    write_3mf(
        out_path=output_path,
        asset=asset,
        profile=profile,
        mapping=mapping,
        triangle_filament_ids=triangle_filament_ids,
        layer_height_mm=layer_height_mm,
        export_mode=export_mode,
    )

    return ConvertResult(
        output_path=output_path,
        geometry_report=geom,
        compat_report=compat,
        mapping_result=mapping,
        metadata={
            "max_colors": max_colors,
            "layer_height_mm": layer_height_mm,
            "palette_size": int(len(palette_list)),
            "input_path": str(asset.original_input_path),
            "selected_mesh_path": str(asset.source_path),
            "target_longest_edge_mm": target_longest_edge_mm,
            "applied_scale_factor": scale_factor,
            "target_faces": target_faces,
            "final_faces": int(asset.faces.shape[0]),
            "label_smoothing_iterations": label_smoothing_iterations,
            "graph_smoothing_iterations": graph_smoothing_iterations,
            "uv_smoothing_iterations": uv_smoothing_iterations,
            "min_island_faces": min_island_faces,
            "paint_safe_remapped_count": int(paint_safe_remapped_count),
            "export_mode": export_mode,
        },
    )
