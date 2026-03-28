# spectrum3mf

Python pipeline for converting textured 3D assets into Snapmaker U1-compatible multicolor 3MF files with palette mapping, virtual mixes, and a Streamlit UI.

## Status

Early-stage project. APIs and output details may change.

## Features

- Convert textured assets to U1-oriented 3MF output.
- Map extracted palette colors to physical filament slots.
- Generate virtual mixed filament definitions.
- Preview conversion and mapping via Streamlit.
- For best results, use .obj + the corresponding .mtl and image files.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## CLI

```bash
u1fs profile list
u1fs convert --input your_model.glb --out out/model_v094.3mf --printer u1 --max-colors 12 --layer-height 0.08
u1fs inspect --input out/model_v094.3mf
```

## Web UI

```bash
streamlit run u1_pipeline_web/app.py
```

## Compatibility Scope

- Baseline target: Snapmaker U1 / FullSpectrum v0.94-style workflows.
- This project is independent and not affiliated with Snapmaker or FullSpectrum.

## License

AGPL-3.0-or-later. See the `LICENSE` file.

