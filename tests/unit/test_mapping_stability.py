from u1_pipeline_core.mix_optimizer import map_palette_to_filaments
from u1_pipeline_core.profile_manager import default_profile


def test_mapping_stable_ids() -> None:
    profile = default_profile()
    palette = [(250, 200, 80), (230, 120, 60), (255, 255, 255)]
    a = map_palette_to_filaments(palette, profile)
    b = map_palette_to_filaments(palette, profile)
    assert a.palette_to_filament_id == b.palette_to_filament_id
    assert [v.filament_id for v in a.virtual_mixes] == [v.filament_id for v in b.virtual_mixes]
