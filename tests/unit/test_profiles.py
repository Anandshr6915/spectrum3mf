from pathlib import Path

from u1_pipeline_core.profile_manager import (
    default_profile,
    profile_to_dict,
    save_profile,
    validate_profile_dict,
)


def test_profile_validation_ok(tmp_path: Path) -> None:
    p = default_profile()
    payload = profile_to_dict(p)
    assert validate_profile_dict(payload) == []


def test_profile_validation_duplicate_slot(tmp_path: Path) -> None:
    p = default_profile()
    payload = profile_to_dict(p)
    payload["slots"][1]["slot_id"] = 1
    errs = validate_profile_dict(payload)
    assert any("duplicate" in e for e in errs)


def test_save_profile(tmp_path: Path) -> None:
    p = default_profile()
    save_profile(tmp_path, p)
    assert (tmp_path / "profiles" / "u1_default.json").exists()
