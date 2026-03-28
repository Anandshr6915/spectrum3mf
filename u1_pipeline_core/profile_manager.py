from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .color_math import hex_to_rgb
from .types import FilamentProfile, MixPolicy, SlotProfile


def profiles_dir(root: Path) -> Path:
    out = root / "profiles"
    out.mkdir(parents=True, exist_ok=True)
    return out


def active_profile_pointer(root: Path) -> Path:
    return profiles_dir(root) / "active_profile.json"


def profile_path(root: Path, name: str) -> Path:
    return profiles_dir(root) / f"{name}.json"


def validate_profile_dict(payload: dict) -> list[str]:
    errors: list[str] = []
    slots = payload.get("slots")
    if not isinstance(slots, list) or len(slots) != 4:
        errors.append("slots must contain exactly four entries")
        return errors

    seen: set[int] = set()
    for slot in slots:
        sid = slot.get("slot_id")
        if sid not in (1, 2, 3, 4):
            errors.append(f"invalid slot_id: {sid}")
            continue
        if sid in seen:
            errors.append(f"duplicate slot_id: {sid}")
        seen.add(sid)
        hex_value = str(slot.get("hex", ""))
        try:
            hex_to_rgb(hex_value)
        except ValueError:
            errors.append(f"invalid hex for slot {sid}: {hex_value}")
    return errors


def profile_from_dict(payload: dict) -> FilamentProfile:
    slots = tuple(
        SlotProfile(
            slot_id=int(slot["slot_id"]),
            hex=str(slot["hex"]).upper(),
            material=str(slot.get("material", "PLA")),
            brand=str(slot.get("brand", "")),
            label=str(slot.get("label", "")),
            td=None if slot.get("td") is None else float(slot.get("td")),
        )
        for slot in sorted(payload["slots"], key=lambda s: int(s["slot_id"]))
    )
    mp = payload.get("mix_policy", {})
    policy = MixPolicy(
        max_virtual_slots=int(mp.get("max_virtual_slots", 40)),
        max_pattern_len=int(mp.get("max_pattern_len", 6)),
        prefer_2_color_mix=bool(mp.get("prefer_2_color_mix", True)),
        allow_3_4_color_pattern=bool(mp.get("allow_3_4_color_pattern", False)),
    )
    return FilamentProfile(
        name=str(payload["name"]),
        version=int(payload.get("version", 1)),
        printer=str(payload.get("printer", "Snapmaker U1")),
        slots=slots,
        mix_policy=policy,
    )


def profile_to_dict(profile: FilamentProfile) -> dict:
    return {
        "name": profile.name,
        "version": profile.version,
        "printer": profile.printer,
        "slots": [asdict(s) for s in profile.slots],
        "mix_policy": asdict(profile.mix_policy),
    }


def save_profile(root: Path, profile: FilamentProfile) -> Path:
    path = profile_path(root, profile.name)
    path.write_text(json.dumps(profile_to_dict(profile), indent=2), encoding="utf-8")
    return path


def load_profile(root: Path, name: str) -> FilamentProfile:
    path = profile_path(root, name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_profile_dict(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return profile_from_dict(payload)


def list_profiles(root: Path) -> list[str]:
    return sorted(p.stem for p in profiles_dir(root).glob("*.json") if p.name != "active_profile.json")


def activate_profile(root: Path, name: str) -> None:
    ptr = active_profile_pointer(root)
    ptr.write_text(json.dumps({"active": name}, indent=2), encoding="utf-8")


def get_active_profile_name(root: Path) -> str | None:
    ptr = active_profile_pointer(root)
    if not ptr.exists():
        return None
    payload = json.loads(ptr.read_text(encoding="utf-8"))
    active = payload.get("active")
    return str(active) if active else None


def default_profile() -> FilamentProfile:
    return FilamentProfile(
        name="u1_default",
        version=1,
        printer="Snapmaker U1",
        slots=(
            SlotProfile(slot_id=1, hex="#FFFFFF", material="PLA", label="T1"),
            SlotProfile(slot_id=2, hex="#FF9900", material="PLA", label="T2"),
            SlotProfile(slot_id=3, hex="#FFCC33", material="PLA", label="T3"),
            SlotProfile(slot_id=4, hex="#FF3333", material="PLA", label="T4"),
        ),
        mix_policy=MixPolicy(),
    )


def ensure_default_profile(root: Path) -> FilamentProfile:
    if "u1_default" not in list_profiles(root):
        save_profile(root, default_profile())
    if get_active_profile_name(root) is None:
        activate_profile(root, "u1_default")
    return load_profile(root, get_active_profile_name(root) or "u1_default")
