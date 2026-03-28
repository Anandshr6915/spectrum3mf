from __future__ import annotations

import numpy as np
from trimesh.remesh import subdivide


def refine_mesh_with_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray | None,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if target_faces <= 0 or faces.shape[0] >= target_faces:
        return vertices, faces, uv

    cur_v = np.asarray(vertices, dtype=np.float64)
    cur_f = np.asarray(faces, dtype=np.int64)
    cur_uv = None if uv is None else np.asarray(uv, dtype=np.float64)

    prev_v = cur_v
    prev_f = cur_f
    prev_uv = cur_uv

    while cur_f.shape[0] < target_faces:
        prev_v, prev_f, prev_uv = cur_v, cur_f, cur_uv
        if cur_uv is not None:
            cur_v, cur_f, attrs = subdivide(cur_v, cur_f, vertex_attributes={"uv": cur_uv})
            cur_uv = np.asarray(attrs.get("uv"), dtype=np.float64) if attrs.get("uv") is not None else None
        else:
            cur_v, cur_f, _attrs = subdivide(cur_v, cur_f, vertex_attributes=None)
            cur_uv = None
        if cur_f.shape[0] > 600000:
            break

    # Keep the level closest to requested target.
    if abs(prev_f.shape[0] - target_faces) < abs(cur_f.shape[0] - target_faces):
        return prev_v, prev_f, prev_uv
    return cur_v, cur_f, cur_uv

