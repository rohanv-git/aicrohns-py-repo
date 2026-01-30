"""
P1: Data Exploration for CD Fibrosis Study
"""

import scanpy as sc
import os

print("="*80)
print("P1: DATA EXPLORATION")
print("="*80)

sc.settings.set_figure_params(dpi=100, facecolor="white")

# Load data
print("\n[1/3] Loading data...")
adata = sc.read_h5ad("/Users/rvellamcheti/Downloads/Cleaned_raw_annotated_object_LK.v2.h5ad")
print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

# Preserve raw counts and log-transform (fixes warning)
if adata.raw is None:
    adata.raw = adata.copy()
    sc.pp.log1p(adata)
    print("Log-transformed data (raw counts preserved)")

# Check CD patients
print("\n[2/3] Analyzing CD patients...")
adata_crohns = adata[adata.obs['disease'].isin(["MONDO_0005011"]), :].copy()

print(f"Total CD patients: {adata_crohns.obs['donor_id'].nunique()}")
print("\nCells per donor:")
print(adata_crohns.obs['donor_id'].value_counts().head(10))

# Generate UMAPs
print("\n[3/3] Generating visualizations...")
os.makedirs("Figures", exist_ok=True)

sc.pl.umap(adata, color="disease", save="_P1_disease.png", show=False)
sc.pl.umap(adata, color="tissue", save="_P1_tissue.png", show=False)
sc.pl.umap(adata, color="organ", save="_P1_organ.png", show=False)
sc.pl.umap(adata, color="annotation2v2", save="_P1_celltypes.png", show=False)
sc.pl.umap(adata, color="species", save="_P1_species.png", show=False)

print("Visualizations saved to figures/")

# Save processed data
adata.write("/Users/rvellamcheti/Downloads/Cleaned_raw_annotated_object_LK.v2.h5ad")
print("Processed data saved")
print("="*80)