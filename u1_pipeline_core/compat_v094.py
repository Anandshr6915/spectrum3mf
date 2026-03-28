from __future__ import annotations

from .types import CompatReport, FilamentProfile, MappingResult


# FullSpectrum installs often support virtual slots well beyond 24 (e.g. 40+).
V094_MAX_FILAMENT_ID = 64


def validate_mapping_v094(mapping: MappingResult, profile: FilamentProfile) -> CompatReport:
    warnings: list[str] = []

    if len(profile.slots) != 4:
        warnings.append("v0.94 baseline expects exactly 4 physical slots")

    ids = list(mapping.palette_to_filament_id)
    for vm in mapping.virtual_mixes:
        ids.append(vm.filament_id)

    if any(i < 1 for i in ids):
        warnings.append("Found invalid filament ID < 1")
    if any(i > V094_MAX_FILAMENT_ID for i in ids):
        warnings.append(f"Found filament ID > {V094_MAX_FILAMENT_ID}")

    if len(mapping.virtual_mixes) > profile.mix_policy.max_virtual_slots:
        warnings.append("Virtual mix count exceeds profile max_virtual_slots")

    warnings.append("Importer optimization is not required; mapping is self-contained for v0.94")
    return CompatReport(ok=not any("invalid" in w.lower() for w in warnings), warnings=warnings)
