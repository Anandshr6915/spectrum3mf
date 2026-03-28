from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .color_math import delta_e_76, rgb_to_lab
from .types import MeshAsset, PaletteExtraction


def _sample_texture(texture: np.ndarray, uv: np.ndarray) -> tuple[int, int, int]:
    h, w, _ = texture.shape
    # Clamp UVs instead of wrapping; wrapping causes seam bleed from unrelated atlas regions.
    u = float(np.clip(uv[0], 0.0, 1.0))
    v = float(np.clip(uv[1], 0.0, 1.0))
    x = min(w - 1, max(0, int(round(u * (w - 1)))))
    y = min(h - 1, max(0, int(round((1.0 - v) * (h - 1)))))
    px = texture[y, x]
    return (int(px[0]), int(px[1]), int(px[2]))


def _triangle_rgb(asset: MeshAsset) -> np.ndarray:
    out = np.zeros((asset.faces.shape[0], 3), dtype=np.uint8)
    if asset.uv is not None and asset.texture_rgb is not None:
        # Denoise high-frequency texture speckles before face sampling.
        tex_img = Image.fromarray(asset.texture_rgb, mode="RGB")
        # Keep detail: too much blur washes out texture ridges and highlights.
        tex_img = tex_img.filter(ImageFilter.GaussianBlur(radius=0.8))
        texture = np.asarray(tex_img, dtype=np.uint8)

        barycentric_samples = np.asarray(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [0.6, 0.2, 0.2],
                [0.2, 0.6, 0.2],
                [0.2, 0.2, 0.6],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        for i, face in enumerate(asset.faces):
            uv_face = asset.uv[np.asarray(face, dtype=np.int64)]
            samples: list[tuple[int, int, int]] = []
            for w in barycentric_samples:
                uv_point = uv_face[0] * w[0] + uv_face[1] * w[1] + uv_face[2] * w[2]
                samples.append(_sample_texture(texture, uv_point))
            # Median is robust to seam leakage/outlier texels while preserving tone.
            out[i] = np.median(np.asarray(samples, dtype=np.float64), axis=0).astype(np.uint8)
        return out

    out[:] = np.array([200, 140, 90], dtype=np.uint8)
    return out


def _kmeans_lab(samples_rgb: np.ndarray, k: int, iterations: int = 18) -> tuple[np.ndarray, np.ndarray]:
    samples_lab = np.array([rgb_to_lab(tuple(int(c) for c in row)) for row in samples_rgb], dtype=np.float64)
    n = samples_lab.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int32)
    k = max(1, min(k, n))

    # Deterministic init: preserve luminance extremes + spread.
    l_vals = samples_lab[:, 0]
    idx_min_l = int(np.argmin(l_vals))
    idx_max_l = int(np.argmax(l_vals))
    centroids = [samples_lab[idx_min_l], samples_lab[idx_max_l]]
    used = {idx_min_l, idx_max_l}
    while len(centroids) < k:
        best_idx = 0
        best_d = -1.0
        for i in range(n):
            if i in used:
                continue
            d = min(np.sum((samples_lab[i] - c) ** 2) for c in centroids)
            if d > best_d:
                best_d = d
                best_idx = i
        used.add(best_idx)
        centroids.append(samples_lab[best_idx])
    cent = np.asarray(centroids, dtype=np.float64)

    assign = np.zeros((n,), dtype=np.int32)
    for _ in range(iterations):
        d2 = np.sum((samples_lab[:, None, :] - cent[None, :, :]) ** 2, axis=2)
        new_assign = np.argmin(d2, axis=1).astype(np.int32)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for ci in range(k):
            mask = assign == ci
            if not np.any(mask):
                continue
            cent[ci] = np.mean(samples_lab[mask], axis=0)

    # Convert centroids back via nearest sample for stable RGB palette.
    palette_rgb = np.zeros((k, 3), dtype=np.uint8)
    for ci in range(k):
        mask = assign == ci
        if not np.any(mask):
            palette_rgb[ci] = samples_rgb[0]
            continue
        idxs = np.where(mask)[0]
        c = cent[ci]
        best = idxs[0]
        best_d = float("inf")
        for i in idxs:
            d = np.sum((samples_lab[i] - c) ** 2)
            if d < best_d:
                best_d = d
                best = i
        palette_rgb[ci] = samples_rgb[best]
    return palette_rgb, assign


def _quantize_palette(triangle_rgb: np.ndarray, max_colors: int) -> tuple[np.ndarray, np.ndarray]:
    palette, tri_idx = _kmeans_lab(triangle_rgb, k=max_colors)
    return palette.astype(np.uint8), tri_idx.astype(np.int32)


def extract_palette(asset: MeshAsset, max_colors: int = 12) -> PaletteExtraction:
    tri_rgb = _triangle_rgb(asset)
    palette, tri_idx = _quantize_palette(tri_rgb, max_colors=max(1, int(max_colors)))
    return PaletteExtraction(
        triangle_rgb=tri_rgb,
        palette_rgb=palette,
        triangle_palette_index=tri_idx,
    )
