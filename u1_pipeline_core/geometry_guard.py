from __future__ import annotations

import numpy as np
import trimesh

from .types import GeometryReport, MeshAsset

U1_BUILD_VOLUME_MM = (270.0, 270.0, 270.0)


def inspect_geometry(asset: MeshAsset) -> GeometryReport:
    warnings: list[str] = []
    if asset.faces.size == 0 or asset.vertices.size == 0:
        warnings.append("Mesh is empty")

    bbox_min = tuple(np.min(asset.vertices, axis=0).tolist())
    bbox_max = tuple(np.max(asset.vertices, axis=0).tolist())
    bbox_size_np = np.asarray(bbox_max) - np.asarray(bbox_min)
    bbox_size = tuple(bbox_size_np.tolist())

    mesh = trimesh.Trimesh(vertices=asset.vertices, faces=asset.faces, process=False)
    try:
        components = len(mesh.split(only_watertight=False))
    except Exception:
        components = 1
    coherent = components == 1
    if not coherent:
        warnings.append(f"Mesh has {components} disconnected components")

    fits = all(float(s) <= lim for s, lim in zip(bbox_size, U1_BUILD_VOLUME_MM))
    if not fits:
        warnings.append(
            f"Mesh bbox {bbox_size} exceeds U1 build volume {U1_BUILD_VOLUME_MM}"
        )

    return GeometryReport(
        coherent=coherent,
        components=components,
        triangle_count=int(asset.faces.shape[0]),
        vertex_count=int(asset.vertices.shape[0]),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_size=bbox_size,
        fits_u1_volume=fits,
        warnings=warnings,
    )
