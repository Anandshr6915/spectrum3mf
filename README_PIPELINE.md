# U1 Pipeline (v0.94-first)

## Install

```bash
python -m pip install -e .
```

## CLI

```bash
u1fs profile list
u1fs convert --input croissant.glb --out out/croissant_v094.3mf --printer u1 --max-colors 12 --layer-height 0.08
u1fs inspect --input out/croissant_v094.3mf
```

## Streamlit

```bash
streamlit run u1_pipeline_web/app.py
```

## Notes

- Baseline target: Snapmaker Orca FullSpectrum v0.94.
- The writer emits a self-contained mapping report and mixed filament definitions metadata.
