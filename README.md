# COMP683 Project Implementation

This repository now contains an end-to-end implementation of the proposal in:

- [proposal/COMP683_Project.pdf](proposal/COMP683_Project.pdf)

It uses the local data in [data](data) and implements:

1. No correction + Leiden
2. Harmony + Leiden
3. BBKNN + Leiden

with evaluation of:

- ARI
- NMI
- Rare-cell recovery rate (default thresholds: 0.5%, 1.0%, 2.0%)

across Leiden resolutions:

- 0.2, 0.5, 1.0, 1.5

## Files Added

- [src/comp683_pipeline.py](src/comp683_pipeline.py): full analysis pipeline
- [environment.yml](environment.yml): reproducible conda environment

## Environment Setup

```bash
conda env create -f environment.yml
conda activate comp683
```

## Run Full Analysis

```bash
python src/comp683_pipeline.py \
  --data-dir data \
  --out-dir outputs
```

## Run Faster Smoke Test

```bash
python src/comp683_pipeline.py \
  --data-dir data \
  --out-dir outputs_smoke \
  --max-cells-per-platform 1200 \
  --n-hvg 1000 \
  --n-pcs 30
```

## Outputs

Pipeline outputs are written under your selected output directory:

- `adata/combined_preprocessed.h5ad`
- `adata/pipeline_baseline_leiden.h5ad`
- `adata/pipeline_harmony_leiden.h5ad`
- `adata/pipeline_bbknn_leiden.h5ad`
- `metrics/metrics_by_pipeline_resolution.csv`
- `metrics/metrics_summary.csv`
- `figures/*` (UMAP triplets, rare-recovery curves, immune case-study plots)

## Notes

- The script automatically restricts analysis to tissues present in both droplet and FACS metadata.
- Droplet expression matrices are loaded from `data/droplet.zip` (10X mtx format).
- FACS expression matrices are loaded from `data/FACS.zip` (*-counts.csv format).
- For large runs, use a machine with sufficient RAM and disk for h5ad outputs.
