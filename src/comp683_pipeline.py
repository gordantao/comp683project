#!/usr/bin/env python3
"""COMP683 project implementation for Tabula Muris rare-cell preservation analysis.

This script implements the proposal end-to-end using local files in data/:
1) Load droplet and FACS expression matrices from zip archives
2) Keep tissues present in both platforms
3) Build a combined AnnData object and preprocess with Scanpy
4) Run three pipelines:
   - No correction + Leiden
   - Harmony + Leiden
   - BBKNN + Leiden
5) Compute ARI/NMI and rare-cell recovery over multiple resolutions
6) Generate figures and save clustered AnnData outputs
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import anndata as ad
import bbknn
import harmonypy as hm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.io import mmread
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _safe_key(value: float) -> str:
    return str(value).replace(".", "_")


def _cluster_key(resolution: float) -> str:
    return f"leiden_r{_safe_key(resolution)}"


def _rare_col(threshold: float) -> str:
    return f"rare_recovery_lt{_safe_key(threshold)}pct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COMP683 Tabula Muris analysis pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing data files")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for outputs")
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 1.0, 1.5],
        help="Leiden resolutions to evaluate",
    )
    parser.add_argument(
        "--rare-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Rare-cell percentage thresholds (strictly less than threshold)",
    )
    parser.add_argument(
        "--fixed-resolution",
        type=float,
        default=1.0,
        help="Resolution used for UMAP/case-study summaries",
    )
    parser.add_argument("--n-hvg", type=int, default=2000, help="Number of highly variable genes")
    parser.add_argument("--n-pcs", type=int, default=50, help="Number of principal components")
    parser.add_argument("--n-neighbors", type=int, default=15, help="Number of neighbors for kNN graph")
    parser.add_argument("--min-genes", type=int, default=200, help="Minimum genes per cell")
    parser.add_argument("--min-cells", type=int, default=3, help="Minimum cells per gene")
    parser.add_argument(
        "--max-cells-per-platform",
        type=int,
        default=None,
        help="Optional cap for cells per platform (useful for fast smoke tests)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation")
    return parser.parse_args()


def _load_metadata(data_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        "annotations_droplet": pd.read_csv(data_dir / "annotations_droplet.csv", low_memory=False),
        "annotations_facs": pd.read_csv(data_dir / "annotations_facs.csv", low_memory=False),
        "metadata_droplet": pd.read_csv(data_dir / "metadata_droplet.csv", low_memory=False),
        "metadata_facs": pd.read_csv(data_dir / "metadata_FACS.csv", low_memory=False),
    }


def _common_tissues(metadata_droplet: pd.DataFrame, metadata_facs: pd.DataFrame) -> List[str]:
    droplet_tissues = set(metadata_droplet["tissue"].dropna().astype(str))
    facs_tissues = set(metadata_facs["tissue"].dropna().astype(str))
    common = sorted(droplet_tissues & facs_tissues)
    if not common:
        raise RuntimeError("No common tissues found between droplet and FACS metadata")
    return common


def _read_text_lines(zf: zipfile.ZipFile, path: str) -> List[str]:
    with zf.open(path) as handle:
        return [line.decode("utf-8").strip() for line in handle if line.strip()]


def _parse_droplet_folder(folder_name: str) -> tuple[str, str]:
    match = re.match(r"^(.*)-(10X_.*)$", folder_name)
    if not match:
        raise ValueError(f"Unexpected droplet folder format: {folder_name}")
    return match.group(1), match.group(2)


def _load_single_droplet_folder(
    zf: zipfile.ZipFile,
    folder_name: str,
) -> ad.AnnData:
    base = f"droplet/{folder_name}"
    matrix_bytes = zf.read(f"{base}/matrix.mtx")
    mtx = mmread(io.BytesIO(matrix_bytes)).tocsr().transpose().tocsr().astype(np.float32)

    genes = [line.split("\t")[0] for line in _read_text_lines(zf, f"{base}/genes.tsv")]
    barcodes_raw = _read_text_lines(zf, f"{base}/barcodes.tsv")

    tissue, channel = _parse_droplet_folder(folder_name)
    barcodes = [bc.rsplit("-", 1)[0] for bc in barcodes_raw]
    obs_names = [f"{channel}_{bc}" for bc in barcodes]

    obs = pd.DataFrame(index=pd.Index(obs_names, name="cell"))
    obs["platform"] = "droplet"
    obs["channel"] = channel
    obs["tissue_from_matrix"] = tissue

    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    adata = ad.AnnData(X=mtx, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def _load_droplet(
    data_dir: Path,
    common_tissues: Sequence[str],
    annotations_droplet: pd.DataFrame,
    metadata_droplet: pd.DataFrame,
) -> ad.AnnData:
    zip_path = data_dir / "droplet.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing droplet archive: {zip_path}")

    adatas: List[ad.AnnData] = []
    with zipfile.ZipFile(zip_path) as zf:
        folder_names = sorted(
            {
                path.split("/")[1]
                for path in zf.namelist()
                if path.startswith("droplet/") and path.endswith("/matrix.mtx")
            }
        )

        for folder_name in folder_names:
            tissue, _ = _parse_droplet_folder(folder_name)
            if tissue not in common_tissues:
                continue
            logging.info("Loading droplet matrix: %s", folder_name)
            adatas.append(_load_single_droplet_folder(zf, folder_name))

    if not adatas:
        raise RuntimeError("No droplet matrices loaded for common tissues")

    droplet = ad.concat(adatas, join="outer", merge="same", index_unique=None)
    droplet.var_names_make_unique()

    ann = annotations_droplet.set_index("cell")
    for col in ann.columns:
        droplet.obs[col] = ann.reindex(droplet.obs_names)[col]

    meta = metadata_droplet.set_index("channel")
    droplet.obs["tissue"] = droplet.obs["channel"].map(meta["tissue"])
    droplet.obs["subtissue"] = droplet.obs["channel"].map(meta.get("subtissue", pd.Series(dtype=object)))

    return droplet


def _load_facs_counts_sparse(
    zf: zipfile.ZipFile,
    csv_path: str,
) -> ad.AnnData:
    with zf.open(csv_path) as handle:
        header = handle.readline().decode("utf-8").rstrip("\r\n")
        cell_names = header.split(",")[1:]

        row_chunks: List[np.ndarray] = []
        col_chunks: List[np.ndarray] = []
        data_chunks: List[np.ndarray] = []
        genes: List[str] = []

        for gene_idx, line_bytes in enumerate(handle):
            line = line_bytes.decode("utf-8").rstrip("\r\n")
            if not line:
                continue
            gene, values_csv = line.split(",", 1)
            values = np.fromstring(values_csv, sep=",", dtype=np.float32)
            if values.shape[0] != len(cell_names):
                raise ValueError(
                    f"Malformed row in {csv_path}: expected {len(cell_names)} values, got {values.shape[0]}"
                )
            nz = np.flatnonzero(values)
            if nz.size > 0:
                row_chunks.append(nz.astype(np.int32, copy=False))
                col_chunks.append(np.full(nz.size, gene_idx, dtype=np.int32))
                data_chunks.append(values[nz])
            genes.append(gene)

    n_cells = len(cell_names)
    n_genes = len(genes)

    if data_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        data = np.concatenate(data_chunks)
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
        data = np.array([], dtype=np.float32)

    X = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes), dtype=np.float32)

    obs = pd.DataFrame(index=pd.Index(cell_names, name="cell"))
    obs["platform"] = "FACS"

    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def _load_facs(
    data_dir: Path,
    common_tissues: Sequence[str],
    annotations_facs: pd.DataFrame,
    metadata_facs: pd.DataFrame,
) -> ad.AnnData:
    zip_path = data_dir / "FACS.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing FACS archive: {zip_path}")

    adatas: List[ad.AnnData] = []
    with zipfile.ZipFile(zip_path) as zf:
        count_files = sorted(
            [
                path
                for path in zf.namelist()
                if path.startswith("FACS/") and path.endswith("-counts.csv")
            ]
        )

        for csv_path in count_files:
            tissue = Path(csv_path).name.replace("-counts.csv", "")
            if tissue not in common_tissues:
                continue
            logging.info("Loading FACS matrix: %s", Path(csv_path).name)
            adata = _load_facs_counts_sparse(zf, csv_path)
            adata.obs["tissue_from_matrix"] = tissue
            adatas.append(adata)

    if not adatas:
        raise RuntimeError("No FACS matrices loaded for common tissues")

    facs = ad.concat(adatas, join="outer", merge="same", index_unique=None)
    facs.var_names_make_unique()

    ann = annotations_facs.set_index("cell")
    for col in ann.columns:
        facs.obs[col] = ann.reindex(facs.obs_names)[col]

    facs.obs["plate.barcode"] = facs.obs_names.to_series().str.split(".").str[1]
    meta = metadata_facs.set_index("plate.barcode")

    facs.obs["tissue"] = facs.obs["plate.barcode"].map(meta["tissue"])
    facs.obs["subtissue"] = facs.obs["plate.barcode"].map(meta.get("subtissue", pd.Series(dtype=object)))

    return facs


def _downsample_per_platform(adata: ad.AnnData, max_cells: int, seed: int) -> ad.AnnData:
    idx_parts: List[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for platform_name in sorted(adata.obs["platform"].unique()):
        mask = np.where(adata.obs["platform"].values == platform_name)[0]
        if mask.size <= max_cells:
            idx_parts.append(mask)
            continue
        selected = np.sort(rng.choice(mask, size=max_cells, replace=False))
        idx_parts.append(selected)

    idx = np.sort(np.concatenate(idx_parts))
    return adata[idx].copy()


def _preprocess(
    adata: ad.AnnData,
    n_hvg: int,
    n_pcs: int,
    min_genes: int,
    min_cells: int,
    seed: int,
) -> ad.AnnData:
    adata = adata.copy()

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, batch_key="platform", flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()

    max_valid_pcs = min(adata.n_obs - 1, adata.n_vars - 1)
    if max_valid_pcs < 2:
        raise RuntimeError(
            "Too few observations/features remain after preprocessing to run PCA. "
            f"n_obs={adata.n_obs}, n_vars={adata.n_vars}"
        )

    n_pcs_eff = min(n_pcs, max_valid_pcs)
    if n_pcs_eff != n_pcs:
        logging.warning("Requested n_pcs=%d, using n_pcs=%d due to data shape", n_pcs, n_pcs_eff)

    sc.pp.pca(adata, n_comps=n_pcs_eff, svd_solver="arpack", random_state=seed)
    return adata


def _run_leiden_over_resolutions(
    adata: ad.AnnData,
    resolutions: Sequence[float],
    seed: int,
) -> None:
    for resolution in resolutions:
        key = _cluster_key(resolution)
        sc.tl.leiden(adata, resolution=resolution, key_added=key, random_state=seed)


def _run_pipeline_baseline(
    adata: ad.AnnData,
    resolutions: Sequence[float],
    n_neighbors: int,
    seed: int,
) -> ad.AnnData:
    out = adata.copy()
    sc.pp.neighbors(out, n_neighbors=n_neighbors, use_rep="X_pca", random_state=seed)
    sc.tl.umap(out, random_state=seed)
    _run_leiden_over_resolutions(out, resolutions, seed)
    return out


def _run_pipeline_harmony(
    adata: ad.AnnData,
    resolutions: Sequence[float],
    n_neighbors: int,
    seed: int,
) -> ad.AnnData:
    out = adata.copy()
    harmony_meta = out.obs[["platform"]].copy()
    harmony_meta["platform"] = harmony_meta["platform"].astype(str).fillna("unknown")

    # Harmonypy 0.2.0 has edge-case bugs when nclust resolves to 1.
    nclust = max(2, int(min(round(out.n_obs / 30.0), 100)))
    harmony_out = hm.run_harmony(
        out.obsm["X_pca"],
        harmony_meta,
        vars_use="platform",
        random_state=seed,
        nclust=nclust,
    )
    out.obsm["X_pca_harmony"] = harmony_out.Z_corr
    sc.pp.neighbors(out, n_neighbors=n_neighbors, use_rep="X_pca_harmony", random_state=seed)
    sc.tl.umap(out, random_state=seed)
    _run_leiden_over_resolutions(out, resolutions, seed)
    return out


def _run_pipeline_bbknn(
    adata: ad.AnnData,
    resolutions: Sequence[float],
    n_pcs: int,
    seed: int,
) -> ad.AnnData:
    out = adata.copy()
    available_pcs = int(out.obsm["X_pca"].shape[1])
    n_pcs_eff = min(n_pcs, available_pcs)
    bbknn.bbknn(out, batch_key="platform", use_rep="X_pca", n_pcs=n_pcs_eff)
    sc.tl.umap(out, random_state=seed)
    _run_leiden_over_resolutions(out, resolutions, seed)
    return out


def _compute_ari_nmi(
    obs: pd.DataFrame,
    cluster_col: str,
    label_col: str = "cell_ontology_class",
) -> tuple[float, float]:
    valid = obs[label_col].notna() & obs[cluster_col].notna()
    if valid.sum() == 0:
        return np.nan, np.nan

    y_true = obs.loc[valid, label_col].astype(str)
    y_pred = obs.loc[valid, cluster_col].astype(str)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return ari, nmi


def _rare_recovery(
    obs: pd.DataFrame,
    cluster_col: str,
    threshold_pct: float,
    purity_cutoff: float = 0.5,
    label_col: str = "cell_ontology_class",
    tissue_col: str = "tissue",
) -> Dict[str, float]:
    df = obs[[label_col, tissue_col, cluster_col]].copy()
    df = df.dropna()

    if df.empty:
        return {"n_rare": 0.0, "n_recovered": 0.0, "rate": np.nan}

    rare_total = 0
    recovered_total = 0
    threshold = threshold_pct / 100.0

    for tissue, tissue_df in df.groupby(tissue_col, observed=False):
        label_freq = tissue_df[label_col].value_counts(normalize=True)
        rare_labels = label_freq[label_freq < threshold].index.tolist()
        if not rare_labels:
            continue

        for label in rare_labels:
            rare_total += 1
            rare_cells = tissue_df[tissue_df[label_col] == label]
            cluster_counts_for_label = rare_cells[cluster_col].value_counts()

            recovered = False
            for cluster_name, count_in_cluster_from_label in cluster_counts_for_label.items():
                full_cluster_size = df[df[cluster_col] == cluster_name].shape[0]
                if full_cluster_size == 0:
                    continue
                purity = count_in_cluster_from_label / full_cluster_size
                if purity >= purity_cutoff:
                    recovered = True
                    break

            if recovered:
                recovered_total += 1

    rate = recovered_total / rare_total if rare_total > 0 else np.nan
    return {"n_rare": float(rare_total), "n_recovered": float(recovered_total), "rate": rate}


def _evaluate_pipeline(
    pipeline_name: str,
    adata: ad.AnnData,
    resolutions: Sequence[float],
    rare_thresholds: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for resolution in resolutions:
        cluster_col = _cluster_key(resolution)
        ari, nmi = _compute_ari_nmi(adata.obs, cluster_col=cluster_col)

        row: Dict[str, float] = {
            "pipeline": pipeline_name,
            "resolution": resolution,
            "n_cells": float(adata.n_obs),
            "n_genes": float(adata.n_vars),
            "ari": ari,
            "nmi": nmi,
        }

        for threshold in rare_thresholds:
            rare_stats = _rare_recovery(adata.obs, cluster_col=cluster_col, threshold_pct=threshold)
            row[_rare_col(threshold)] = rare_stats["rate"]
            row[f"n_rare_lt{_safe_key(threshold)}pct"] = rare_stats["n_rare"]
            row[f"n_recovered_lt{_safe_key(threshold)}pct"] = rare_stats["n_recovered"]

        rows.append(row)

    return pd.DataFrame(rows)


def _save_umap_triplet(adata: ad.AnnData, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sc.pl.umap(adata, color="platform", ax=axes[0], show=False)
    sc.pl.umap(adata, color="tissue", ax=axes[1], show=False)
    sc.pl.umap(adata, color="cell_ontology_class", ax=axes[2], show=False, legend_loc="right margin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_rare_recovery_plots(metrics_df: pd.DataFrame, rare_thresholds: Sequence[float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for threshold in rare_thresholds:
        col = _rare_col(threshold)
        if col not in metrics_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for pipeline_name, part in metrics_df.groupby("pipeline"):
            part = part.sort_values("resolution")
            ax.plot(part["resolution"], part[col], marker="o", label=pipeline_name)

        ax.set_title(f"Rare-cell recovery (<{threshold}% per tissue)")
        ax.set_xlabel("Leiden resolution")
        ax.set_ylabel("Recovery rate")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"rare_recovery_lt{_safe_key(threshold)}pct.png", dpi=220)
        plt.close(fig)


def _save_immune_case_study(
    adata: ad.AnnData,
    fixed_resolution: float,
    out_dir: Path,
    pipeline_name: str,
) -> None:
    cluster_col = _cluster_key(fixed_resolution)
    labels = adata.obs["cell_ontology_class"].astype(str)
    immune_mask = labels.str.contains("T cell|macrophage", case=False, na=False)

    if immune_mask.sum() == 0:
        logging.warning("No immune-case-study cells found for %s", pipeline_name)
        return

    immune = adata[immune_mask].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.umap(immune, color="cell_ontology_class", ax=axes[0], show=False, legend_loc="right margin")
    sc.pl.umap(immune, color="tissue", ax=axes[1], show=False, legend_loc="right margin")
    fig.tight_layout()
    fig.savefig(out_dir / f"{pipeline_name}_immune_umap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    immune_table = (
        immune.obs.groupby(["cell_ontology_class", "platform", "tissue", cluster_col], observed=False)
        .size()
        .reset_index(name="n_cells")
    )
    immune_table.to_csv(out_dir / f"{pipeline_name}_immune_cluster_table.csv", index=False)


def _sanitize_obs_for_h5ad(adata: ad.AnnData) -> ad.AnnData:
    out = adata.copy()
    for col in out.obs.columns:
        if pd.api.types.is_object_dtype(out.obs[col]):
            out.obs[col] = out.obs[col].astype(str)
    return out


def main() -> None:
    _setup_logging()
    args = _parse_args()

    np.random.seed(args.seed)

    data_dir = args.data_dir
    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    metrics_dir = out_dir / "metrics"
    adata_dir = out_dir / "adata"

    for p in [fig_dir, metrics_dir, adata_dir]:
        p.mkdir(parents=True, exist_ok=True)

    logging.info("Loading metadata")
    metadata = _load_metadata(data_dir)
    common_tissues = _common_tissues(metadata["metadata_droplet"], metadata["metadata_facs"])
    logging.info("Common tissues (%d): %s", len(common_tissues), ", ".join(common_tissues))

    logging.info("Loading droplet expression matrices")
    droplet = _load_droplet(
        data_dir,
        common_tissues,
        metadata["annotations_droplet"],
        metadata["metadata_droplet"],
    )

    logging.info("Loading FACS expression matrices")
    facs = _load_facs(
        data_dir,
        common_tissues,
        metadata["annotations_facs"],
        metadata["metadata_facs"],
    )

    logging.info("Droplet shape before concat: %s", droplet.shape)
    logging.info("FACS shape before concat: %s", facs.shape)

    combined = ad.concat([droplet, facs], join="inner", merge="same", index_unique="__")
    combined.obs["platform"] = combined.obs["platform"].astype(str)

    if args.max_cells_per_platform is not None:
        logging.info("Applying max-cells-per-platform cap: %d", args.max_cells_per_platform)
        combined = _downsample_per_platform(combined, args.max_cells_per_platform, args.seed)

    logging.info("Combined shape (common genes): %s", combined.shape)

    logging.info("Preprocessing combined data")
    combined_pp = _preprocess(
        combined,
        n_hvg=args.n_hvg,
        n_pcs=args.n_pcs,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
        seed=args.seed,
    )
    _sanitize_obs_for_h5ad(combined_pp).write_h5ad(adata_dir / "combined_preprocessed.h5ad", compression="gzip")

    logging.info("Running baseline pipeline")
    baseline = _run_pipeline_baseline(
        combined_pp,
        resolutions=args.resolutions,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
    )

    logging.info("Running harmony pipeline")
    harmony = _run_pipeline_harmony(
        combined_pp,
        resolutions=args.resolutions,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
    )

    logging.info("Running BBKNN pipeline")
    bbknn_adata = _run_pipeline_bbknn(
        combined_pp,
        resolutions=args.resolutions,
        n_pcs=args.n_pcs,
        seed=args.seed,
    )

    _sanitize_obs_for_h5ad(baseline).write_h5ad(adata_dir / "pipeline_baseline_leiden.h5ad", compression="gzip")
    _sanitize_obs_for_h5ad(harmony).write_h5ad(adata_dir / "pipeline_harmony_leiden.h5ad", compression="gzip")
    _sanitize_obs_for_h5ad(bbknn_adata).write_h5ad(adata_dir / "pipeline_bbknn_leiden.h5ad", compression="gzip")

    logging.info("Evaluating metrics")
    metrics_df = pd.concat(
        [
            _evaluate_pipeline("baseline", baseline, args.resolutions, args.rare_thresholds),
            _evaluate_pipeline("harmony", harmony, args.resolutions, args.rare_thresholds),
            _evaluate_pipeline("bbknn", bbknn_adata, args.resolutions, args.rare_thresholds),
        ],
        ignore_index=True,
    )
    metrics_df.to_csv(metrics_dir / "metrics_by_pipeline_resolution.csv", index=False)

    summary_cols = ["pipeline", "resolution", "ari", "nmi"] + [_rare_col(t) for t in args.rare_thresholds]
    summary_df = metrics_df[summary_cols].copy()
    summary_df.to_csv(metrics_dir / "metrics_summary.csv", index=False)

    if not args.skip_plots:
        logging.info("Saving UMAP figures and rare-recovery curves")
        _save_umap_triplet(baseline, fig_dir / "baseline_umap_triplet.png")
        _save_umap_triplet(harmony, fig_dir / "harmony_umap_triplet.png")
        _save_umap_triplet(bbknn_adata, fig_dir / "bbknn_umap_triplet.png")

        _save_rare_recovery_plots(metrics_df, args.rare_thresholds, fig_dir)

        _save_immune_case_study(baseline, args.fixed_resolution, fig_dir, "baseline")
        _save_immune_case_study(harmony, args.fixed_resolution, fig_dir, "harmony")
        _save_immune_case_study(bbknn_adata, args.fixed_resolution, fig_dir, "bbknn")

    logging.info("Done. Outputs saved in %s", out_dir)


if __name__ == "__main__":
    main()
