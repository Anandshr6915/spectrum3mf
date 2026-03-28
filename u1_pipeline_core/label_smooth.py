from __future__ import annotations

from collections import Counter
import math

import numpy as np
import trimesh
from .color_math import delta_e_76, rgb_to_lab


def smooth_face_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    iterations: int = 2,
) -> np.ndarray:
    if iterations <= 0 or labels.size == 0:
        return labels

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    adjacency = mesh.face_adjacency
    if adjacency.size == 0:
        return labels

    neighbors: list[list[int]] = [[] for _ in range(len(faces))]
    for a, b in adjacency:
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))

    current = labels.astype(np.int32).copy()
    for _ in range(iterations):
        updated = current.copy()
        for i in range(len(current)):
            nbs = neighbors[i]
            if not nbs:
                continue
            counts = Counter(int(current[n]) for n in nbs)
            majority_label, majority_count = counts.most_common(1)[0]
            # Change only if there is a clear local majority.
            if majority_count >= 3 and majority_label != int(current[i]):
                updated[i] = majority_label
        current = updated
    return current


def smooth_face_labels_edge_aware(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    face_rgb: np.ndarray,
    iterations: int = 2,
    color_edge_delta_e: float = 14.0,
) -> np.ndarray:
    if iterations <= 0 or labels.size == 0:
        return labels

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    adjacency = mesh.face_adjacency
    if adjacency.size == 0:
        return labels

    face_lab = np.array([rgb_to_lab((int(r), int(g), int(b))) for r, g, b in face_rgb], dtype=float)

    neighbors: list[list[int]] = [[] for _ in range(len(faces))]
    for a, b in adjacency:
        a_i = int(a)
        b_i = int(b)
        de = delta_e_76(tuple(face_lab[a_i]), tuple(face_lab[b_i]))
        if de <= color_edge_delta_e:
            neighbors[a_i].append(b_i)
            neighbors[b_i].append(a_i)

    current = labels.astype(np.int32).copy()
    for _ in range(iterations):
        updated = current.copy()
        for i in range(len(current)):
            nbs = neighbors[i]
            if not nbs:
                continue
            counts = Counter(int(current[n]) for n in nbs)
            majority_label, majority_count = counts.most_common(1)[0]
            if majority_count >= 3 and majority_label != int(current[i]):
                updated[i] = majority_label
        current = updated
    return current


def remove_small_label_islands(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    face_rgb: np.ndarray | None = None,
    min_component_faces: int = 120,
    iterations: int = 2,
    color_edge_delta_e: float = 16.0,
) -> np.ndarray:
    if min_component_faces <= 1 or labels.size == 0:
        return labels

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    adjacency = mesh.face_adjacency
    if adjacency.size == 0:
        return labels

    neighbors: list[list[int]] = [[] for _ in range(len(faces))]
    for a, b in adjacency:
        a_i = int(a)
        b_i = int(b)
        if face_rgb is not None:
            lab_a = rgb_to_lab((int(face_rgb[a_i][0]), int(face_rgb[a_i][1]), int(face_rgb[a_i][2])))
            lab_b = rgb_to_lab((int(face_rgb[b_i][0]), int(face_rgb[b_i][1]), int(face_rgb[b_i][2])))
            if delta_e_76(lab_a, lab_b) > color_edge_delta_e:
                continue
        neighbors[a_i].append(b_i)
        neighbors[b_i].append(a_i)

    cur = labels.astype(np.int32).copy()
    for _ in range(max(1, iterations)):
        visited = np.zeros((len(cur),), dtype=bool)
        changed = False
        for start in range(len(cur)):
            if visited[start]:
                continue
            label = int(cur[start])
            stack = [start]
            comp: list[int] = []
            visited[start] = True
            while stack:
                i = stack.pop()
                comp.append(i)
                for nb in neighbors[i]:
                    if visited[nb] or int(cur[nb]) != label:
                        continue
                    visited[nb] = True
                    stack.append(nb)

            if len(comp) >= min_component_faces:
                continue

            border = Counter()
            for i in comp:
                for nb in neighbors[i]:
                    nb_label = int(cur[nb])
                    if nb_label != label:
                        border[nb_label] += 1
            if not border:
                continue
            new_label = border.most_common(1)[0][0]
            for i in comp:
                cur[i] = new_label
            changed = True
        if not changed:
            break
    return cur


def suppress_thin_label_chains(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    *,
    min_same_neighbors: int = 2,
    iterations: int = 2,
) -> np.ndarray:
    if labels.size == 0 or iterations <= 0:
        return labels

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    adjacency = mesh.face_adjacency
    if adjacency.size == 0:
        return labels

    neighbors: list[list[int]] = [[] for _ in range(len(faces))]
    for a, b in adjacency:
        a_i = int(a)
        b_i = int(b)
        neighbors[a_i].append(b_i)
        neighbors[b_i].append(a_i)

    cur = labels.astype(np.int32).copy()
    for _ in range(iterations):
        changed = False
        updated = cur.copy()
        for i in range(len(cur)):
            nbs = neighbors[i]
            if len(nbs) < 2:
                continue
            cur_label = int(cur[i])
            same = 0
            border = Counter()
            for nb in nbs:
                nb_label = int(cur[nb])
                if nb_label == cur_label:
                    same += 1
                else:
                    border[nb_label] += 1
            # If a face has too little support from same-label neighbors, treat it as
            # a thin chain pixel and absorb into dominant local label.
            if same >= int(min_same_neighbors) or not border:
                continue
            new_label, cnt = border.most_common(1)[0]
            if cnt < 2:
                continue
            updated[i] = int(new_label)
            changed = True
        cur = updated
        if not changed:
            break
    return cur


def smooth_face_labels_uv_aware(
    faces: np.ndarray,
    uv: np.ndarray | None,
    labels: np.ndarray,
    face_rgb: np.ndarray,
    *,
    iterations: int = 2,
    uv_radius: float = 0.02,
    color_edge_delta_e: float = 12.0,
) -> np.ndarray:
    if uv is None or labels.size == 0 or iterations <= 0:
        return labels
    if faces.size == 0 or uv.shape[0] == 0:
        return labels

    uv_cent = uv[np.asarray(faces, dtype=np.int64)].mean(axis=1)
    uv_cent = np.mod(uv_cent, 1.0)
    face_lab = np.asarray([rgb_to_lab((int(r), int(g), int(b))) for r, g, b in face_rgb], dtype=np.float64)

    cell = max(1e-4, float(uv_radius))
    inv_cell = 1.0 / cell
    grid: dict[tuple[int, int], list[int]] = {}
    for i, p in enumerate(uv_cent):
        key = (int(math.floor(float(p[0]) * inv_cell)), int(math.floor(float(p[1]) * inv_cell)))
        grid.setdefault(key, []).append(i)

    cur = labels.astype(np.int32).copy()
    radius2 = uv_radius * uv_radius
    sigma2 = max(1e-8, (uv_radius * 0.65) ** 2)

    for _ in range(iterations):
        updated = cur.copy()
        changed = False
        for i, p in enumerate(uv_cent):
            cx = int(math.floor(float(p[0]) * inv_cell))
            cy = int(math.floor(float(p[1]) * inv_cell))
            weights: dict[int, float] = {}
            self_label = int(cur[i])
            self_w = 0.0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in grid.get((cx + dx, cy + dy), []):
                        dp = uv_cent[j] - p
                        d2 = float(dp[0] * dp[0] + dp[1] * dp[1])
                        if d2 > radius2:
                            continue
                        de = delta_e_76(tuple(face_lab[i]), tuple(face_lab[j]))
                        if de > color_edge_delta_e:
                            continue
                        w = float(np.exp(-d2 / (2.0 * sigma2)))
                        lj = int(cur[j])
                        weights[lj] = weights.get(lj, 0.0) + w
                        if lj == self_label:
                            self_w += w
            if not weights:
                continue
            best_label = max(weights.items(), key=lambda kv: kv[1])[0]
            best_w = float(weights[best_label])
            # Require a clear weighted majority to avoid eroding true features.
            if best_label != self_label and best_w > (1.35 * max(1e-9, self_w)):
                updated[i] = int(best_label)
                changed = True
        cur = updated
        if not changed:
            break
    return cur


def optimize_labels_graph_energy(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    face_rgb: np.ndarray,
    label_rgb_by_id: dict[int, tuple[int, int, int]],
    *,
    iterations: int = 4,
    smooth_weight: float = 1.6,
    confidence_margin: float = 8.0,
) -> np.ndarray:
    if iterations <= 0 or labels.size == 0 or not label_rgb_by_id:
        return labels

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    adjacency = mesh.face_adjacency
    if adjacency.size == 0:
        return labels

    face_lab = np.asarray([rgb_to_lab((int(r), int(g), int(b))) for r, g, b in face_rgb], dtype=np.float64)

    label_ids = sorted(int(k) for k in label_rgb_by_id.keys())
    label_lab = np.asarray([rgb_to_lab(label_rgb_by_id[k]) for k in label_ids], dtype=np.float64)

    # Data term for each face/label pair.
    data_cost = np.sum((face_lab[:, None, :] - label_lab[None, :, :]) ** 2, axis=2) ** 0.5
    best_idx = np.argmin(data_cost, axis=1)
    part = np.partition(data_cost, kth=1 if data_cost.shape[1] > 1 else 0, axis=1)
    if data_cost.shape[1] > 1:
        confidence = part[:, 1] - part[:, 0]
    else:
        confidence = np.full((data_cost.shape[0],), 999.0, dtype=np.float64)

    id_to_idx = {fid: i for i, fid in enumerate(label_ids)}
    cur = labels.astype(np.int32).copy()

    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(len(faces))]
    # Encourage smooth labels on low-angle + low-color-gradient areas only.
    adj_angles = mesh.face_adjacency_angles
    for e, (a, b) in enumerate(adjacency):
        a_i = int(a)
        b_i = int(b)
        de = delta_e_76(tuple(face_lab[a_i]), tuple(face_lab[b_i]))
        angle_deg = float(np.degrees(float(adj_angles[e]))) if e < len(adj_angles) else 0.0
        color_w = float(np.exp(-de / 9.0))
        angle_w = float(np.exp(-angle_deg / 16.0))
        w = smooth_weight * color_w * angle_w
        if w <= 0.03:
            continue
        neighbors[a_i].append((b_i, w))
        neighbors[b_i].append((a_i, w))

    for _ in range(iterations):
        changed = False
        updated = cur.copy()
        for i in range(len(cur)):
            nbs = neighbors[i]
            if not nbs:
                continue
            # Keep high-confidence assignments stable to preserve sharp texture features.
            if confidence[i] >= confidence_margin:
                continue

            candidate_labels = {int(cur[i])}
            for nb, _w in nbs:
                candidate_labels.add(int(cur[nb]))

            best_label = int(cur[i])
            cur_idx = id_to_idx.get(best_label)
            if cur_idx is None:
                continue
            best_energy = float(data_cost[i, cur_idx])
            for nb, w in nbs:
                if int(cur[nb]) != best_label:
                    best_energy += w

            for cand in candidate_labels:
                cand_idx = id_to_idx.get(int(cand))
                if cand_idx is None:
                    continue
                e_data = float(data_cost[i, cand_idx])
                e_smooth = 0.0
                for nb, w in nbs:
                    if int(cur[nb]) != int(cand):
                        e_smooth += w
                e_total = e_data + e_smooth
                if e_total + 1e-9 < best_energy:
                    best_energy = e_total
                    best_label = int(cand)
            if best_label != int(cur[i]):
                updated[i] = best_label
                changed = True
        cur = updated
        if not changed:
            break
    return cur
