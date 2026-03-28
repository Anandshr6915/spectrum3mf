from pathlib import Path

from u1_pipeline_core.conversion import convert_asset_to_3mf
from u1_pipeline_core.profile_manager import default_profile
from u1_pipeline_core.threemf_writer import inspect_3mf


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_convert_croissant_glb(tmp_path: Path) -> None:
    src = _root() / "croissant.glb"
    if not src.exists():
        return
    out = tmp_path / "croissant_glb.3mf"
    result = convert_asset_to_3mf(
        input_path=src,
        profile=default_profile(),
        output_path=out,
        max_colors=10,
        layer_height_mm=0.08,
    )
    assert result.geometry_report.triangle_count > 0
    rep = inspect_3mf(out)
    assert rep["ok"]


def test_convert_croissant_obj(tmp_path: Path) -> None:
    src = _root() / "Croissant.obj"
    if not src.exists():
        return
    out = tmp_path / "croissant_obj.3mf"
    result = convert_asset_to_3mf(
        input_path=src,
        profile=default_profile(),
        output_path=out,
        max_colors=10,
        layer_height_mm=0.08,
    )
    assert result.geometry_report.triangle_count > 0
    rep = inspect_3mf(out)
    assert rep["ok"]
