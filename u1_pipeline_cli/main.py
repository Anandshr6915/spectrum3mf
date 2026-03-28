from __future__ import annotations

import argparse
import json
from pathlib import Path

from u1_pipeline_core.conversion import convert_asset_to_3mf
from u1_pipeline_core.presets import PRESETS, get_preset
from u1_pipeline_core.profile_manager import (
    activate_profile,
    ensure_default_profile,
    get_active_profile_name,
    list_profiles,
    load_profile,
    profile_to_dict,
    save_profile,
    validate_profile_dict,
)
from u1_pipeline_core.threemf_writer import inspect_3mf


def _root() -> Path:
    return Path.cwd()


def cmd_profile(args: argparse.Namespace) -> int:
    root = _root()
    ensure_default_profile(root)

    if args.profile_cmd == "list":
        active = get_active_profile_name(root)
        for name in list_profiles(root):
            marker = "*" if name == active else " "
            print(f"{marker} {name}")
        return 0

    if args.profile_cmd == "validate":
        p = Path(args.path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        errors = validate_profile_dict(payload)
        if errors:
            for err in errors:
                print(f"ERROR: {err}")
            return 2
        print("OK")
        return 0

    if args.profile_cmd == "create":
        payload = {
            "name": args.name,
            "version": 1,
            "printer": "Snapmaker U1",
            "slots": [
                {"slot_id": 1, "hex": "#FFFFFF", "material": "PLA", "brand": "", "label": "T1", "td": None},
                {"slot_id": 2, "hex": "#FF9900", "material": "PLA", "brand": "", "label": "T2", "td": None},
                {"slot_id": 3, "hex": "#FFCC33", "material": "PLA", "brand": "", "label": "T3", "td": None},
                {"slot_id": 4, "hex": "#FF3333", "material": "PLA", "brand": "", "label": "T4", "td": None},
            ],
            "mix_policy": {
                "max_virtual_slots": 40,
                "max_pattern_len": 6,
                "prefer_2_color_mix": True,
                "allow_3_4_color_pattern": False,
            },
        }
        out = root / "profiles" / f"{args.name}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(str(out))
        return 0

    if args.profile_cmd == "clone":
        src = load_profile(root, args.source)
        payload = profile_to_dict(src)
        payload["name"] = args.target
        out = root / "profiles" / f"{args.target}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(str(out))
        return 0

    if args.profile_cmd == "activate":
        activate_profile(root, args.name)
        print(f"active={args.name}")
        return 0

    return 1


def cmd_convert(args: argparse.Namespace) -> int:
    root = _root()
    ensure_default_profile(root)
    profile_name = args.profile or get_active_profile_name(root) or "u1_default"
    profile = load_profile(root, profile_name)
    preset = get_preset(args.preset)

    max_colors = int(args.max_colors) if args.max_colors is not None else preset.max_colors
    target_faces = int(args.target_faces) if args.target_faces is not None else preset.target_faces
    label_smoothing = (
        int(args.label_smoothing) if args.label_smoothing is not None else preset.label_smoothing
    )
    graph_smoothing = (
        int(args.graph_smoothing) if args.graph_smoothing is not None else preset.graph_smoothing
    )
    uv_smoothing = int(args.uv_smoothing) if args.uv_smoothing is not None else 2
    min_island_faces = (
        int(args.min_island_faces) if args.min_island_faces is not None else preset.min_island_faces
    )

    result = convert_asset_to_3mf(
        input_path=Path(args.input),
        profile=profile,
        output_path=Path(args.out),
        max_colors=max_colors,
        layer_height_mm=args.layer_height,
        target_longest_edge_mm=args.target_size_mm,
        target_faces=target_faces,
        label_smoothing_iterations=label_smoothing,
        min_island_faces=min_island_faces,
        graph_smoothing_iterations=graph_smoothing,
        uv_smoothing_iterations=uv_smoothing,
        export_mode=args.export_mode,
    )

    print(json.dumps(
        {
            "output": str(result.output_path),
            "source_selection": {
                "input_path": result.metadata.get("input_path"),
                "selected_mesh_path": result.metadata.get("selected_mesh_path"),
                "preset": args.preset,
                "target_longest_edge_mm": result.metadata.get("target_longest_edge_mm"),
                "applied_scale_factor": result.metadata.get("applied_scale_factor"),
                "target_faces": result.metadata.get("target_faces"),
                "final_faces": result.metadata.get("final_faces"),
                "label_smoothing_iterations": result.metadata.get("label_smoothing_iterations"),
                "graph_smoothing_iterations": result.metadata.get("graph_smoothing_iterations"),
                "uv_smoothing_iterations": result.metadata.get("uv_smoothing_iterations"),
                "min_island_faces": result.metadata.get("min_island_faces"),
                "paint_safe_remapped_count": result.metadata.get("paint_safe_remapped_count"),
                "export_mode": result.metadata.get("export_mode"),
            },
            "geometry": {
                "coherent": result.geometry_report.coherent,
                "components": result.geometry_report.components,
                "triangle_count": result.geometry_report.triangle_count,
                "bbox_size": result.geometry_report.bbox_size,
                "fits_u1_volume": result.geometry_report.fits_u1_volume,
                "warnings": result.geometry_report.warnings,
            },
            "compat": {
                "ok": result.compat_report.ok,
                "warnings": result.compat_report.warnings,
            },
            "virtual_mixes": len(result.mapping_result.virtual_mixes),
        },
        indent=2,
    ))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    report = inspect_3mf(Path(args.input))
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="u1fs")
    sub = p.add_subparsers(dest="cmd", required=True)

    prof = sub.add_parser("profile")
    prof_sub = prof.add_subparsers(dest="profile_cmd", required=True)

    prof_sub.add_parser("list")
    c = prof_sub.add_parser("create")
    c.add_argument("name")
    cl = prof_sub.add_parser("clone")
    cl.add_argument("source")
    cl.add_argument("target")
    a = prof_sub.add_parser("activate")
    a.add_argument("name")
    v = prof_sub.add_parser("validate")
    v.add_argument("path")

    conv = sub.add_parser("convert")
    conv.add_argument("--input", required=True)
    conv.add_argument("--profile", default=None)
    conv.add_argument("--out", required=True)
    conv.add_argument("--printer", default="u1")
    conv.add_argument("--preset", choices=sorted(PRESETS.keys()), default="balanced")
    conv.add_argument("--max-colors", type=int, default=None)
    conv.add_argument("--layer-height", type=float, default=0.08)
    conv.add_argument("--target-size-mm", type=float, default=80.0)
    conv.add_argument("--target-faces", type=int, default=None)
    conv.add_argument("--label-smoothing", type=int, default=None)
    conv.add_argument("--graph-smoothing", type=int, default=None)
    conv.add_argument("--uv-smoothing", type=int, default=None)
    conv.add_argument("--min-island-faces", type=int, default=None)
    conv.add_argument("--export-mode", choices=["paint", "materials", "region-split"], default="paint")

    ins = sub.add_parser("inspect")
    ins.add_argument("--input", required=True)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "profile":
        raise SystemExit(cmd_profile(args))
    if args.cmd == "convert":
        raise SystemExit(cmd_convert(args))
    if args.cmd == "inspect":
        raise SystemExit(cmd_inspect(args))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
