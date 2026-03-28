from u1_pipeline_core.color_math import blend_weighted_srgb, delta_e_76, rgb_to_lab


def test_delta_e_zero_for_equal_colors() -> None:
    lab = rgb_to_lab((100, 120, 140))
    assert delta_e_76(lab, lab) == 0.0


def test_blend_is_deterministic() -> None:
    a = blend_weighted_srgb([(255, 255, 255), (255, 0, 0)], [1, 1])
    b = blend_weighted_srgb([(255, 255, 255), (255, 0, 0)], [1, 1])
    assert a == b
