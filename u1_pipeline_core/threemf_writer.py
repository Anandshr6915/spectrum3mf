from __future__ import annotations

import json
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape

import numpy as np

from .color_math import rgb_to_hex
from .types import FilamentProfile, MappingResult, MeshAsset


def _build_mixed_definitions(mapping: MappingResult) -> str:
    # FullSpectrum expected CSV-ish row format (semicolon-separated rows):
    # a,b,enabled,custom,mix_b,pointillism_flag,pattern[,gIDS][,wWEIGHTS][,mMODE][,dDELETED][,oORIGIN_AUTO][,uSTABLE_ID]
    # See MixedFilamentManager::parse_row_definition / load_custom_entries.
    chunks: list[str] = []
    for i, vm in enumerate(mapping.virtual_mixes):
        if len(vm.components) < 2:
            continue
        a = int(vm.components[0])
        b = int(vm.components[1])
        # mix_b is percentage for component_b.
        mix_b = int(vm.weights_percent[1]) if len(vm.weights_percent) >= 2 else 50
        pattern = vm.pattern if vm.pattern else "12"
        gradient_ids = "".join(str(c) for c in vm.components)
        gradient_weights = "/".join(str(int(w)) for w in vm.weights_percent)
        stable_id = 100000 + i
        row_tokens = [
            str(a),
            str(b),
            "1",  # enabled
            "1",  # custom
            str(mix_b),
            "0",  # legacy pointillism flag off
            pattern,
            f"g{gradient_ids}",
            f"w{gradient_weights}",
            "m2",  # Simple distribution mode
            "d0",
            "o0",
            f"u{stable_id}",
        ]
        chunks.append(",".join(row_tokens))
    return ";".join(chunks)


def _submesh_from_face_indices(
    vertices: np.ndarray, faces: np.ndarray, face_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sel_faces = faces[face_indices]
    unique_vertices, inverse = np.unique(sel_faces.reshape(-1), return_inverse=True)
    local_vertices = vertices[unique_vertices]
    local_faces = inverse.reshape((-1, 3))
    return local_vertices, local_faces


def _paint_color_code(filament_id: int) -> str:
    if filament_id <= 0:
        return "0"
    if filament_id == 1:
        return "4"
    if filament_id == 2:
        return "8"
    if filament_id >= 3:
        # Matches Orca/Bambu paint_color no-split pattern used in community tooling.
        value = ((filament_id - 3) << 4) | 0xC
        return format(value, "X")
    return "0"


def _model_xml_region_split(
    asset: MeshAsset, triangle_filament_ids: np.ndarray, display_hex_by_id: dict[int, str]
) -> str:
    base_entries = []
    max_id = max(display_hex_by_id)
    for idx in range(1, max_id + 1):
        hx = display_hex_by_id.get(idx, "#808080")
        base_entries.append(f'<base name="F{idx}" displaycolor="{escape(hx)}"/>')
    bases = "\n      ".join(base_entries)

    objects_xml: list[str] = []
    build_items: list[str] = []
    object_id = 1
    unique_filament_ids = sorted(set(int(i) for i in triangle_filament_ids.tolist()))

    for filament_id in unique_filament_ids:
        face_indices = np.where(triangle_filament_ids == filament_id)[0]
        if face_indices.size == 0:
            continue
        local_vertices, local_faces = _submesh_from_face_indices(
            asset.vertices, asset.faces, face_indices
        )
        vertex_rows = []
        for v in local_vertices:
            vertex_rows.append(f'<vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>')
        vertices_xml = "\n          ".join(vertex_rows)

        tri_rows = []
        p = int(filament_id) - 1
        for f in local_faces:
            tri_rows.append(
                f'<triangle v1="{int(f[0])}" v2="{int(f[1])}" v3="{int(f[2])}" pid="1" p1="{p}" p2="{p}" p3="{p}"/>'
            )
        triangles_xml = "\n          ".join(tri_rows)
        objects_xml.append(
            f'''<object id="{object_id}" type="model" pid="1" pindex="{p}">
      <mesh>
        <vertices>
          {vertices_xml}
        </vertices>
        <triangles>
          {triangles_xml}
        </triangles>
      </mesh>
    </object>'''
        )
        build_items.append(f'<item objectid="{object_id}"/>')
        object_id += 1

    objects_blob = "\n    ".join(objects_xml)
    build_blob = "\n    ".join(build_items)

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <basematerials id="1">
      {bases}
    </basematerials>
    {objects_blob}
  </resources>
  <build>
    {build_blob}
  </build>
</model>
'''


def _model_xml_paint(asset: MeshAsset, triangle_filament_ids: np.ndarray) -> str:
    vertex_rows = []
    for v in asset.vertices:
        vertex_rows.append(f'<vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>')
    vertices_xml = "\n        ".join(vertex_rows)

    tri_rows = []
    for i, f in enumerate(asset.faces):
        code = _paint_color_code(int(triangle_filament_ids[i]))
        tri_rows.append(
            f'<triangle v1="{int(f[0])}" v2="{int(f[1])}" v3="{int(f[2])}" paint_color="{code}"/>'
        )
    triangles_xml = "\n        ".join(tri_rows)

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>
        {vertices_xml}
        </vertices>
        <triangles>
        {triangles_xml}
        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1"/>
  </build>
</model>
'''


def _model_xml_material_paint(
    asset: MeshAsset,
    triangle_filament_ids: np.ndarray,
    display_hex_by_id: dict[int, str],
) -> str:
    max_id = max(int(i) for i in triangle_filament_ids.tolist()) if triangle_filament_ids.size else 1
    base_entries = []
    for idx in range(1, max_id + 1):
        hx = display_hex_by_id.get(idx, "#808080")
        base_entries.append(f'<base name="F{idx}" displaycolor="{escape(hx)}"/>')
    bases = "\n      ".join(base_entries)

    vertex_rows = []
    for v in asset.vertices:
        vertex_rows.append(f'<vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>')
    vertices_xml = "\n        ".join(vertex_rows)

    tri_rows = []
    for i, f in enumerate(asset.faces):
        p = int(max(1, int(triangle_filament_ids[i]))) - 1
        tri_rows.append(
            f'<triangle v1="{int(f[0])}" v2="{int(f[1])}" v3="{int(f[2])}" pid="1" p1="{p}" p2="{p}" p3="{p}"/>'
        )
    triangles_xml = "\n        ".join(tri_rows)

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <basematerials id="1">
      {bases}
    </basematerials>
    <object id="1" type="model" pid="1" pindex="0">
      <mesh>
        <vertices>
        {vertices_xml}
        </vertices>
        <triangles>
        {triangles_xml}
        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1"/>
  </build>
</model>
'''


def write_3mf(
    out_path: Path,
    asset: MeshAsset,
    profile: FilamentProfile,
    mapping: MappingResult,
    triangle_filament_ids: np.ndarray,
    layer_height_mm: float,
    export_mode: str = "paint",
) -> Path:
    display_hex_by_id: dict[int, str] = {s.slot_id: s.hex.upper() for s in profile.slots}
    for vm in mapping.virtual_mixes:
        display_hex_by_id[vm.filament_id] = vm.display_hex

    if export_mode == "region-split":
        model_xml = _model_xml_region_split(asset, triangle_filament_ids, display_hex_by_id)
    elif export_mode == "materials":
        model_xml = _model_xml_material_paint(asset, triangle_filament_ids, display_hex_by_id)
    else:
        model_xml = _model_xml_paint(asset, triangle_filament_ids)
    content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
  <Default Extension="config" ContentType="application/json"/>
</Types>
'''
    rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
'''

    project_settings = {
        "version": "u1-pipeline-0.1",
        "printer": profile.printer,
        "filament_colour": [s.hex.upper() for s in profile.slots],
        "mixed_filament_definitions": _build_mixed_definitions(mapping),
        "layer_height": layer_height_mm,
        "base_support_filament": 1,
        "u1_export_mode": export_mode,
    }
    mapping_report = {
        "palette_to_filament_id": mapping.palette_to_filament_id,
        "palette_to_display_hex": mapping.palette_to_display_hex,
        "virtual_mixes": [
            {
                "filament_id": vm.filament_id,
                "display_hex": vm.display_hex,
                "components": list(vm.components),
                "weights_percent": list(vm.weights_percent),
                "pattern": vm.pattern,
            }
            for vm in mapping.virtual_mixes
        ],
        "triangle_count": int(asset.faces.shape[0]),
        "object_count": int(len(set(int(i) for i in triangle_filament_ids.tolist()))),
        "triangle_count_per_filament": {
            str(fid): int(np.sum(triangle_filament_ids == fid))
            for fid in sorted(set(int(i) for i in triangle_filament_ids.tolist()))
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", model_xml)
        zf.writestr("Metadata/project_settings.config", json.dumps(project_settings, indent=2))
        zf.writestr("Metadata/u1_mapping_report.json", json.dumps(mapping_report, indent=2))

    return out_path


def inspect_3mf(path: Path) -> dict:
    out: dict = {"path": str(path), "ok": False, "errors": [], "summary": {}}
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        required = {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"}
        missing = sorted(required - names)
        if missing:
            out["errors"].append(f"Missing required entries: {missing}")
            return out

        model_xml = zf.read("3D/3dmodel.model").decode("utf-8", errors="replace")
        tri_count = model_xml.count("<triangle ")
        vertex_count = model_xml.count("<vertex ")
        out["summary"]["triangles"] = tri_count
        out["summary"]["vertices"] = vertex_count

        if "Metadata/project_settings.config" in names:
            cfg = json.loads(zf.read("Metadata/project_settings.config").decode("utf-8"))
            out["summary"]["filament_colour"] = cfg.get("filament_colour", [])
            out["summary"]["mixed_filament_definitions"] = cfg.get("mixed_filament_definitions", "")
        out["ok"] = tri_count > 0 and vertex_count > 0
    return out
