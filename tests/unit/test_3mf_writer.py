from pathlib import Path

import numpy as np

from u1_pipeline_core.profile_manager import default_profile
from u1_pipeline_core.threemf_writer import inspect_3mf, write_3mf
from u1_pipeline_core.types import MappingResult, MeshAsset


def test_3mf_contains_required_entries(tmp_path: Path) -> None:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    asset = MeshAsset(
        source_path=Path("dummy.obj"),
        original_input_path=Path("dummy.obj"),
        vertices=vertices,
        faces=faces,
        uv=None,
        texture_rgb=None,
    )
    mapping = MappingResult(
        palette_to_filament_id=[1],
        palette_to_display_hex=["#FFFFFF"],
        virtual_mixes=[],
    )
    out = tmp_path / "t.3mf"
    write_3mf(out, asset, default_profile(), mapping, np.array([1], dtype=int), 0.08)
    rep = inspect_3mf(out)
    assert rep["ok"]
    assert rep["summary"]["triangles"] == 1
