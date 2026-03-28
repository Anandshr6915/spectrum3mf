from __future__ import annotations

import json
import os
from pathlib import Path
import sys

# Allow running `python app.py` from inside `u1_pipeline_web/`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import numpy as np

from u1_pipeline_core.conversion import convert_asset_to_3mf
from u1_pipeline_core.asset_ingest import load_asset
from u1_pipeline_core.color_extract import extract_palette
from u1_pipeline_core.color_math import hex_to_rgb
from u1_pipeline_core.label_smooth import (
    optimize_labels_graph_energy,
    remove_small_label_islands,
    smooth_face_labels_edge_aware,
    smooth_face_labels_uv_aware,
)
from u1_pipeline_core.mesh_refine import refine_mesh_with_uv
from u1_pipeline_core.mix_optimizer import assign_colors_to_filaments, map_palette_to_filaments
from u1_pipeline_core.presets import PRESETS, get_preset
from u1_pipeline_core.profile_manager import (
    activate_profile,
    ensure_default_profile,
    get_active_profile_name,
    list_profiles,
    load_profile,
    profile_to_dict,
    save_profile,
)


def _root() -> Path:
    # Keep project-relative paths stable no matter where streamlit is launched from.
    return _ROOT


def _scale_vertices_to_target_longest_edge(vertices: np.ndarray, target_longest_edge_mm: float) -> np.ndarray:
    if target_longest_edge_mm <= 0:
        return vertices
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    size = maxs - mins
    longest = float(np.max(size))
    if longest <= 0:
        return vertices
    return vertices * (float(target_longest_edge_mm) / longest)


def _build_preview(
    *,
    input_path: Path,
    profile_name: str,
    max_colors: int,
    target_size_mm: float,
    target_faces: int,
    label_smoothing: int,
    graph_smoothing: int,
    uv_smoothing: int,
    min_island_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profile = load_profile(_root(), profile_name)
    asset = load_asset(input_path)
    asset.vertices = _scale_vertices_to_target_longest_edge(asset.vertices, float(target_size_mm))
    rv, rf, ru = refine_mesh_with_uv(
        vertices=asset.vertices,
        faces=asset.faces,
        uv=asset.uv,
        target_faces=int(target_faces),
    )
    asset.vertices = rv
    asset.faces = rf
    asset.uv = ru

    palette = extract_palette(asset, max_colors=int(max_colors))
    palette_list = [tuple(int(c) for c in row) for row in palette.palette_rgb]
    mapping = map_palette_to_filaments(palette_list, profile)

    labels = np.asarray(
        assign_colors_to_filaments(
            [tuple(int(c) for c in row) for row in palette.triangle_rgb],
            profile,
            mapping,
        ),
        dtype=np.int32,
    )
    labels = smooth_face_labels_edge_aware(
        vertices=asset.vertices,
        faces=asset.faces,
        labels=labels,
        face_rgb=palette.triangle_rgb,
        iterations=int(label_smoothing),
        color_edge_delta_e=14.0,
    )
    label_rgb_by_id: dict[int, tuple[int, int, int]] = {
        int(slot.slot_id): hex_to_rgb(slot.hex) for slot in profile.slots
    }
    for vm in mapping.virtual_mixes:
        label_rgb_by_id[int(vm.filament_id)] = hex_to_rgb(vm.display_hex)
    labels = optimize_labels_graph_energy(
        vertices=asset.vertices,
        faces=asset.faces,
        labels=labels,
        face_rgb=palette.triangle_rgb,
        label_rgb_by_id=label_rgb_by_id,
        iterations=int(graph_smoothing),
        smooth_weight=2.4,
        confidence_margin=9.0,
    )
    labels = smooth_face_labels_uv_aware(
        faces=asset.faces,
        uv=asset.uv,
        labels=labels,
        face_rgb=palette.triangle_rgb,
        iterations=int(uv_smoothing),
        uv_radius=0.018,
        color_edge_delta_e=11.0,
    )
    labels = remove_small_label_islands(
        vertices=asset.vertices,
        faces=asset.faces,
        labels=labels,
        face_rgb=palette.triangle_rgb,
        min_component_faces=int(min_island_faces),
        iterations=2,
        color_edge_delta_e=16.0,
    )
    face_colors = np.asarray([label_rgb_by_id[int(fid)] for fid in labels], dtype=np.uint8)
    return asset.vertices, asset.faces, face_colors


def _profile_editor(root: Path) -> str:
    ensure_default_profile(root)
    names = list_profiles(root)
    active = get_active_profile_name(root) or "u1_default"
    selected = st.selectbox("Profile", names, index=names.index(active) if active in names else 0)

    if st.button("Set Active Profile"):
        activate_profile(root, selected)
        st.success(f"Active profile: {selected}")

    prof = load_profile(root, selected)
    payload = profile_to_dict(prof)

    st.subheader("Edit Slots")
    for slot in payload["slots"]:
        sid = slot["slot_id"]
        cols = st.columns(5)
        slot["hex"] = cols[0].text_input(f"T{sid} Hex", value=slot["hex"], key=f"hex_{sid}")
        slot["material"] = cols[1].text_input(f"T{sid} Material", value=slot["material"], key=f"mat_{sid}")
        slot["brand"] = cols[2].text_input(f"T{sid} Brand", value=slot["brand"], key=f"brand_{sid}")
        slot["label"] = cols[3].text_input(f"T{sid} Label", value=slot["label"], key=f"label_{sid}")
        td_str = "" if slot.get("td") is None else str(slot.get("td"))
        td_input = cols[4].text_input(f"T{sid} TD", value=td_str, key=f"td_{sid}")
        slot["td"] = None if td_input.strip() == "" else float(td_input)

    if st.button("Save Profile"):
        # Persist edited JSON payload directly; load validation happens on next use.
        (root / "profiles" / f"{selected}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        st.success("Profile saved")

    return selected


def _list_local_assets(root: Path) -> list[str]:
    exts = {".glb", ".obj", ".fbx", ".stl", ".ply", ".3mf", ".usdz"}
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "out", "profiles"}
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if p.suffix.lower() in exts:
                found.append(str(p.relative_to(root)))
    return sorted(found)


def _materialize_uploaded_files(root: Path, uploaded_files: list) -> list[Path]:
    upload_dir = root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for f in uploaded_files:
        out = upload_dir / Path(f.name).name
        out.write_bytes(f.getbuffer())
        written.append(out)
    return written


def run() -> None:
    st.set_page_config(page_title="U1 Pipeline v0.94", layout="wide")
    st.title("Snapmaker U1 FullSpectrum v0.94 Pipeline")
    root = _root()

    st.markdown("Step 1: Choose profile and asset")
    profile_name = _profile_editor(root)

    src_mode = st.radio("Input source", options=["Select local file", "Upload file(s)"], horizontal=True)
    resolved_input_path: Path | None = None
    if src_mode == "Select local file":
        local_assets = _list_local_assets(root)
        default_idx = local_assets.index("croissant.glb") if "croissant.glb" in local_assets else 0
        if local_assets:
            picked = st.selectbox("Input asset", options=local_assets, index=default_idx)
            resolved_input_path = root / picked
        else:
            st.warning("No local asset files found. Use upload mode.")
    else:
        uploaded = st.file_uploader(
            "Upload model/texture files (.glb, .obj+.mtl+textures, .fbx, ...)",
            accept_multiple_files=True,
            type=["glb", "obj", "mtl", "png", "jpg", "jpeg", "fbx", "usdz", "stl", "ply"],
        )
        if uploaded:
            written = _materialize_uploaded_files(root, uploaded)
            preferred = [p for p in written if p.suffix.lower() in (".glb", ".obj", ".fbx", ".stl", ".ply", ".usdz")]
            options = preferred if preferred else written
            picked_upload = st.selectbox("Primary input file", options=[str(p.name) for p in options], index=0)
            resolved_input_path = next(p for p in options if p.name == picked_upload)
            st.caption(f"Saved uploads to: {root / 'uploads'}")

    out_dir_options = ["out", "."]
    out_dir = st.selectbox("Save folder", options=out_dir_options, index=0)
    out_name = st.text_input("Output filename", value="croissant_v094.3mf")
    if not out_name.lower().endswith(".3mf"):
        out_name = f"{out_name}.3mf"
    out_path = (root / out_dir / out_name).resolve()
    preset_name = st.selectbox("Preset", options=sorted(PRESETS.keys()), index=0)
    preset = get_preset(preset_name)
    if st.button("Apply Preset Values"):
        st.session_state["max_colors"] = int(preset.max_colors)
        st.session_state["target_faces"] = int(preset.target_faces)
        st.session_state["label_smoothing"] = int(preset.label_smoothing)
        st.session_state["graph_smoothing"] = int(preset.graph_smoothing)
        st.session_state["uv_smoothing"] = 2
        st.session_state["min_island_faces"] = int(preset.min_island_faces)

    if "max_colors" not in st.session_state:
        st.session_state["max_colors"] = int(preset.max_colors)
    if "target_faces" not in st.session_state:
        st.session_state["target_faces"] = int(preset.target_faces)
    if "label_smoothing" not in st.session_state:
        st.session_state["label_smoothing"] = int(preset.label_smoothing)
    if "graph_smoothing" not in st.session_state:
        st.session_state["graph_smoothing"] = int(preset.graph_smoothing)
    if "uv_smoothing" not in st.session_state:
        st.session_state["uv_smoothing"] = 2
    if "min_island_faces" not in st.session_state:
        st.session_state["min_island_faces"] = int(preset.min_island_faces)

    max_colors = st.slider("Max palette colors", 4, 40, key="max_colors")
    layer_height = st.number_input("Layer height (mm)", min_value=0.04, max_value=0.30, value=0.08, step=0.01)
    target_size = st.number_input("Target longest edge (mm)", min_value=10.0, max_value=250.0, value=80.0, step=5.0)
    target_faces = st.number_input("Target face count", min_value=2000, max_value=200000, step=2000, key="target_faces")
    smoothing = st.slider("Label smoothing iterations", 0, 12, key="label_smoothing")
    graph_smoothing = st.slider("Graph smoothing iterations", 0, 12, key="graph_smoothing")
    uv_smoothing = st.slider("UV smoothing iterations", 0, 8, key="uv_smoothing")
    min_island = st.number_input("Minimum island faces", min_value=5, max_value=2000, step=5, key="min_island_faces")
    export_mode = st.selectbox("Export mode", ["paint", "materials", "region-split"], index=0)

    if st.button("Preview Mapping"):
        if resolved_input_path is None:
            st.error("Please select or upload an input file first.")
            return
        try:
            vertices, faces, face_colors = _build_preview(
                input_path=resolved_input_path,
                profile_name=profile_name,
                max_colors=int(max_colors),
                target_size_mm=float(target_size),
                target_faces=int(target_faces),
                label_smoothing=int(smoothing),
                graph_smoothing=int(graph_smoothing),
                uv_smoothing=int(uv_smoothing),
                min_island_faces=int(min_island),
            )
            try:
                import plotly.graph_objects as go

                mesh = go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    facecolor=[f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in face_colors],
                    flatshading=True,
                    lighting={"ambient": 0.9, "diffuse": 0.2, "specular": 0.0, "roughness": 1.0},
                    lightposition={"x": 0, "y": 0, "z": 1000},
                    showscale=False,
                )
                fig = go.Figure(data=[mesh])
                fig.update_layout(
                    scene_aspectmode="data",
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=640,
                )
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Preview renderer error: {e}")
        except Exception as e:
            st.error(f"Preview pipeline error: {e}")

    if st.button("Convert"):
        if resolved_input_path is None:
            st.error("Please select or upload an input file first.")
            return
        profile = load_profile(root, profile_name)
        result = convert_asset_to_3mf(
            input_path=resolved_input_path,
            profile=profile,
            output_path=Path(out_path),
            max_colors=max_colors,
            layer_height_mm=float(layer_height),
            target_longest_edge_mm=float(target_size),
            target_faces=int(target_faces),
            label_smoothing_iterations=int(smoothing),
            graph_smoothing_iterations=int(graph_smoothing),
            uv_smoothing_iterations=int(uv_smoothing),
            min_island_faces=int(min_island),
            export_mode=str(export_mode),
        )
        st.success(f"Wrote {result.output_path}")
        out_bytes = Path(result.output_path).read_bytes()
        st.download_button(
            "Download .3mf",
            data=out_bytes,
            file_name=Path(result.output_path).name,
            mime="application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
        )
        st.json(
            {
                "geometry": {
                    "coherent": result.geometry_report.coherent,
                    "components": result.geometry_report.components,
                    "triangle_count": result.geometry_report.triangle_count,
                    "bbox_size": result.geometry_report.bbox_size,
                    "fits_u1_volume": result.geometry_report.fits_u1_volume,
                },
                "compat": {
                    "ok": result.compat_report.ok,
                    "warnings": result.compat_report.warnings,
                },
                "virtual_mixes": [vm.__dict__ for vm in result.mapping_result.virtual_mixes],
            }
        )


if __name__ == "__main__":
    run()
