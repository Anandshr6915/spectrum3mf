from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from .types import MeshAsset


def _scene_to_single_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    meshes: list[trimesh.Trimesh] = []
    for geom in scene.geometry.values():
        if isinstance(geom, trimesh.Trimesh):
            meshes.append(geom)
    if not meshes:
        raise ValueError("No mesh geometry found in scene")
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _fallback_texture(path: Path) -> np.ndarray | None:
    candidates = [
        path.with_name("CroissantAlbedo.png"),
        path.with_name("glTF").joinpath("textures", "Croissant_baseColor.png"),
    ]
    for c in candidates:
        if c.exists():
            return np.asarray(Image.open(c).convert("RGB"), dtype=np.uint8)
    return None


def _load_raw_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False, maintain_order=True)
    mesh = _scene_to_single_mesh(loaded) if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input file did not load as mesh")
    return mesh


def _mesh_non_manifold_edges(mesh: trimesh.Trimesh) -> int:
    if mesh.faces.size == 0:
        return 0
    faces = np.asarray(mesh.faces, dtype=np.int64)
    all_edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    all_edges = np.sort(all_edges, axis=1)
    unique, counts = np.unique(all_edges, axis=0, return_counts=True)
    _ = unique
    return int(np.sum(counts > 2))


def _mesh_quality_score(mesh: trimesh.Trimesh) -> tuple[int, int, int]:
    components = len(mesh.split(only_watertight=False))
    non_manifold_edges = _mesh_non_manifold_edges(mesh)
    not_watertight = 0 if mesh.is_watertight else 1
    return (components - 1, non_manifold_edges, not_watertight)


def _select_best_mesh_path(path: Path) -> Path:
    # If user inputs GLB and a sibling OBJ exists, prefer the mesh with better
    # printability metrics (coherence/manifoldness/watertightness).
    if path.suffix.lower() != ".glb":
        return path

    candidates = [path]
    obj_candidates = [path.with_suffix(".obj"), path.parent / "Croissant.obj"]
    for cand in obj_candidates:
        if cand.exists() and cand not in candidates:
            candidates.append(cand)

    best_path = path
    best_score: tuple[int, int, int] | None = None
    for cand in candidates:
        try:
            score = _mesh_quality_score(_load_raw_mesh(cand))
        except Exception:
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_path = cand
    return best_path


def load_asset(path: Path) -> MeshAsset:
    selected_path = _select_best_mesh_path(path)
    mesh = _load_raw_mesh(selected_path)

    uv = None
    texture_rgb = None
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    if getattr(mesh.visual, "material", None) is not None:
        img = getattr(mesh.visual.material, "image", None)
        if img is not None:
            texture_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

    if texture_rgb is None:
        texture_rgb = _fallback_texture(selected_path)

    return MeshAsset(
        source_path=selected_path,
        original_input_path=path,
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        uv=uv,
        texture_rgb=texture_rgb,
    )
