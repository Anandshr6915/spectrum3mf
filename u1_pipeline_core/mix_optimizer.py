from __future__ import annotations

from dataclasses import dataclass
import itertools
import colorsys

from .color_math import blend_weighted_srgb, delta_e_76, hex_to_rgb, rgb_to_hex, rgb_to_lab
from .types import FilamentProfile, MappingResult, VirtualMixDefinition


@dataclass(frozen=True)
class _Candidate:
    key: tuple[int, ...]
    components: tuple[int, ...]
    weights: tuple[int, ...]
    display_rgb: tuple[int, int, int]
    pattern: str


def _hue_sat(rgb: tuple[int, int, int]) -> tuple[float, float]:
    h, s, _v = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    return h * 360.0, s


def _hue_distance_deg(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 360.0 - d)


def perceptual_cost(target_rgb: tuple[int, int, int], candidate_rgb: tuple[int, int, int]) -> float:
    t_lab = rgb_to_lab(target_rgb)
    c_lab = rgb_to_lab(candidate_rgb)
    de = delta_e_76(t_lab, c_lab)
    t_l = t_lab[0]
    c_l = c_lab[0]
    # Keep lightness structure (highlights/shadows) from source textures.
    l_penalty = abs(t_l - c_l)
    t_chroma = (t_lab[1] ** 2 + t_lab[2] ** 2) ** 0.5
    c_chroma = (c_lab[1] ** 2 + c_lab[2] ** 2) ** 0.5
    chroma_penalty = abs(t_chroma - c_chroma)
    # Strongly discourage saturated candidates for near-neutral target colors.
    neutral_penalty = 0.0
    if t_chroma < 16.0 and c_chroma > t_chroma:
        neutral_penalty += 0.85 * (c_chroma - t_chroma)
    if t_chroma < 10.0 and c_chroma > 22.0:
        neutral_penalty += 0.55 * (c_chroma - 22.0)
    t_h, t_s = _hue_sat(target_rgb)
    c_h, _ = _hue_sat(candidate_rgb)
    # Generic hue-consistency term: preserve original texture hue when saturated.
    hue_penalty = (_hue_distance_deg(t_h, c_h) / 180.0) * (18.0 * t_s)
    # Allow hue drift in darker areas where luminance structure matters more than hue.
    hue_penalty *= (0.35 + 0.65 * (t_l / 100.0))
    # Strongly discourage mapping dark regions to overly bright mixes.
    shadow_penalty = 0.0
    if c_l > t_l + 5.0:
        shadow_penalty += 1.2 * (c_l - (t_l + 5.0))
    # Also discourage brightening when target is already dark.
    if t_l < 40.0 and c_l > t_l:
        shadow_penalty += 0.9 * (c_l - t_l)
    # Generic anti-washout term: avoid systematically brightening textured ridges.
    if c_l > t_l:
        ridge_boost = 1.2 if t_l < 65.0 else 1.0
        shadow_penalty += 0.55 * ridge_boost * (c_l - t_l)
    return de + 0.7 * l_penalty + 0.16 * chroma_penalty + hue_penalty + shadow_penalty + neutral_penalty


def _weights_to_percent(weights: tuple[int, ...]) -> tuple[int, ...]:
    total = sum(weights)
    if total <= 0:
        return tuple(0 for _ in weights)
    raw = [100.0 * w / total for w in weights]
    flo = [int(x) for x in raw]
    remainder = 100 - sum(flo)
    frac_order = sorted(range(len(weights)), key=lambda i: raw[i] - flo[i], reverse=True)
    for i in frac_order[:remainder]:
        flo[i] += 1
    return tuple(flo)


def _build_pattern(components: tuple[int, ...], weights: tuple[int, ...]) -> str:
    seq: list[str] = []
    for comp, w in zip(components, weights):
        seq.extend([str(comp)] * int(w))
    return "".join(seq)


def _generate_virtual_candidates(profile: FilamentProfile) -> list[_Candidate]:
    slots = [s.slot_id for s in profile.slots]
    slot_rgb = {s.slot_id: hex_to_rgb(s.hex) for s in profile.slots}
    max_len = max(2, profile.mix_policy.max_pattern_len)
    out: list[_Candidate] = []

    for length in range(2, max_len + 1):
        for a, b in itertools.combinations(slots, 2):
            for wa in range(1, length):
                wb = length - wa
                comps = (a, b)
                weights = (wa, wb)
                out.append(
                    _Candidate(
                        key=(a, b, wa, wb),
                        components=comps,
                        weights=weights,
                        display_rgb=blend_weighted_srgb([slot_rgb[a], slot_rgb[b]], [wa, wb]),
                        pattern=_build_pattern(comps, weights),
                    )
                )

    if profile.mix_policy.allow_3_4_color_pattern:
        for length in range(3, max_len + 1):
            for comps in itertools.combinations(slots, 3):
                for wa in range(1, length - 1):
                    for wb in range(1, length - wa):
                        wc = length - wa - wb
                        weights = (wa, wb, wc)
                        out.append(
                            _Candidate(
                                key=(*comps, *weights),
                                components=comps,
                                weights=weights,
                                display_rgb=blend_weighted_srgb([slot_rgb[c] for c in comps], list(weights)),
                                pattern=_build_pattern(comps, weights),
                            )
                        )
    out.sort(key=lambda c: c.key)
    return out


def map_palette_to_filaments(
    palette_rgb: list[tuple[int, int, int]],
    profile: FilamentProfile,
) -> MappingResult:
    physical = [(s.slot_id, hex_to_rgb(s.hex)) for s in profile.slots]
    virtual_candidates = _generate_virtual_candidates(profile)

    # 1) Build baseline physical-only nearest distances.
    physical_best_dist: list[float] = []
    for rgb in palette_rgb:
        best_dist = min(perceptual_cost(rgb, srgb) for _, srgb in physical)
        physical_best_dist.append(best_dist)

    # 2) Score candidates by total improvement across palette.
    def hue_deg(rgb: tuple[int, int, int]) -> float:
        h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        _ = v
        return float(h * 360.0), float(s)

    target_hues: list[float] = []
    red_like_count = 0
    for rgb in palette_rgb:
        h, s = hue_deg(rgb)
        if s > 0.12:
            target_hues.append(h)
            if h < 20.0 or h > 340.0:
                red_like_count += 1
    red_ratio = (red_like_count / len(target_hues)) if target_hues else 0.0

    scored_candidates: list[tuple[float, _Candidate]] = []
    for cand in virtual_candidates:
        total_improvement = 0.0
        for i, rgb in enumerate(palette_rgb):
            cand_dist = perceptual_cost(rgb, cand.display_rgb)
            total_improvement += max(0.0, physical_best_dist[i] - cand_dist)
        # Penalize pink-ish white+red mixes unless target palette is truly red-heavy.
        comps = tuple(sorted(cand.components))
        if comps == (1, 4) and red_ratio < 0.18:
            total_improvement *= 0.45
        if total_improvement > 0.0:
            scored_candidates.append((total_improvement, cand))
    scored_candidates.sort(key=lambda x: (-x[0], x[1].key))

    # Deduplicate near-equivalent recipes by pair+percent tuple.
    deduped: list[tuple[float, _Candidate]] = []
    seen_recipe_keys: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for score, cand in scored_candidates:
        recipe_key = (cand.components, _weights_to_percent(cand.weights))
        if recipe_key in seen_recipe_keys:
            continue
        seen_recipe_keys.add(recipe_key)
        deduped.append((score, cand))

    max_virtual = max(0, int(profile.mix_policy.max_virtual_slots))
    selected: list[_Candidate] = []

    # Pass 1: pick best candidate per component pair for diversity.
    best_per_pair: dict[tuple[int, ...], tuple[float, _Candidate]] = {}
    for score, cand in deduped:
        pair_key = tuple(sorted(cand.components))
        if pair_key not in best_per_pair:
            best_per_pair[pair_key] = (score, cand)
    diverse = sorted(best_per_pair.values(), key=lambda x: (-x[0], x[1].key))
    for _, cand in diverse:
        if len(selected) >= max_virtual:
            break
        selected.append(cand)

    # Pass 2: fill remaining slots by global score.
    if len(selected) < max_virtual:
        selected_keys = {c.key for c in selected}
        for _, cand in deduped:
            if cand.key in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(cand.key)
            if len(selected) >= max_virtual:
                break

    virtual_defs: list[VirtualMixDefinition] = []
    selected_by_key: dict[tuple[int, ...], VirtualMixDefinition] = {}
    next_virtual_id = 5
    for cand in selected:
        vm = VirtualMixDefinition(
            filament_id=next_virtual_id,
            display_hex=rgb_to_hex(cand.display_rgb),
            components=cand.components,
            weights_percent=_weights_to_percent(cand.weights),
            pattern=cand.pattern,
        )
        selected_by_key[cand.key] = vm
        virtual_defs.append(vm)
        next_virtual_id += 1

    # 3) Map palette only against physical + selected virtual slots.
    palette_to_id: list[int] = []
    palette_to_hex: list[str] = []
    for rgb in palette_rgb:
        best_id = physical[0][0]
        best_hex = rgb_to_hex(physical[0][1])
        best_dist = float("inf")

        for sid, srgb in physical:
            dist = perceptual_cost(rgb, srgb)
            if dist < best_dist:
                best_dist = dist
                best_id = sid
                best_hex = rgb_to_hex(srgb)

        for vm in virtual_defs:
            vm_rgb = hex_to_rgb(vm.display_hex)
            dist = perceptual_cost(rgb, vm_rgb)
            if dist < best_dist:
                best_dist = dist
                best_id = vm.filament_id
                best_hex = vm.display_hex

        palette_to_id.append(best_id)
        palette_to_hex.append(best_hex)

    virtual_mixes = sorted(virtual_defs, key=lambda v: v.filament_id)
    return MappingResult(
        palette_to_filament_id=palette_to_id,
        palette_to_display_hex=palette_to_hex,
        virtual_mixes=virtual_mixes,
    )


def assign_colors_to_filaments(
    colors_rgb: list[tuple[int, int, int]],
    profile: FilamentProfile,
    mapping: MappingResult,
) -> list[int]:
    candidates: list[tuple[int, tuple[int, int, int]]] = [
        (int(slot.slot_id), hex_to_rgb(slot.hex)) for slot in profile.slots
    ]
    candidates.extend((int(vm.filament_id), hex_to_rgb(vm.display_hex)) for vm in mapping.virtual_mixes)

    out: list[int] = []
    for rgb in colors_rgb:
        best_id = candidates[0][0]
        best_cost = float("inf")
        for fid, cand_rgb in candidates:
            c = perceptual_cost(rgb, cand_rgb)
            if c < best_cost:
                best_cost = c
                best_id = fid
        out.append(int(best_id))
    return out
