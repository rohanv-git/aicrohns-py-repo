# P3_colon.py - Create patient embeddings from P2 pathway data - COLON
# WITH FULL CHECKPOINTING

import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Setup
print("="*80)
print("P3: COLON PATIENT PATHWAY PROFILE GENERATION")
print("="*80)

# Create organized directory structure
os.makedirs("checkpoints/colon", exist_ok=True)
os.makedirs("figures/P2_figs/colon", exist_ok=True)
os.makedirs("figures/P3_figs/colon", exist_ok=True)
os.makedirs("figures/P4_figs/colon", exist_ok=True)
os.makedirs("results/P3_results/colon", exist_ok=True)
os.makedirs("results/P4_results/colon", exist_ok=True)

print("\n✓ Directory structure created:")
print("  figures/")
print("    ├── P2_figs/colon/")
print("    ├── P3_figs/colon/")
print("    └── P4_figs/colon/")
print("  results/")
print("    ├── P3_results/colon/")
print("    └── P4_results/colon/")

# File paths - INPUTS (from P2)
P2_GSEA_FILE = "checkpoints/colon/P2_gsea_results_BIG_DARK_ONLY.csv"
P2_H5AD_FILE = "checkpoints/colon/P2_filtered_data.h5ad"

# File paths - OUTPUTS (P3 generates these)
OUTPUT_EMBEDDINGS = "results/P3_results/colon/P3_patient_embeddings.csv"
OUTPUT_GENE_EMBEDDINGS = "results/P3_results/colon/P3_gene_embeddings.csv"
OUTPUT_PATHWAY_SCORES = "results/P3_results/colon/P3_pathway_scores.csv"

# Checkpoint files (intermediate)
CHECKPOINT_PATHWAY_GENES = "checkpoints/colon/P3_pathway_genes.pkl"
CHECKPOINT_GENE_PATHWAY_MATRIX = "checkpoints/colon/P3_gene_pathway_matrix.npz"
CHECKPOINT_COMMON_GENES = "checkpoints/colon/P3_common_genes.pkl"

# Figure paths
FIG_PATIENT_CLUSTERING = "figures/P3_figs/colon/P3_patient_clustering.png"
FIG_PATHWAY_HEATMAP = "figures/P3_figs/colon/P3_pathway_heatmap.png"
FIG_EMBEDDING_DISTRIBUTION = "figures/P3_figs/colon/P3_embedding_distribution.png"
FIG_GENE_EMBEDDING_PCA = "figures/P3_figs/colon/P3_gene_embedding_pca.png"

# ============================================================================
# STEP 1: Load P2 GSEA results
# ============================================================================
print("\n[1/6] Loading P2 GSEA results...")
gsea_df = pd.read_csv(P2_GSEA_FILE)

# Remove infinite NES values
n_inf = np.isinf(gsea_df['nes']).sum()
if n_inf > 0:
    print(f"      ⚠️  Found {n_inf} infinite NES values - removing...")
    gsea_df = gsea_df[~np.isinf(gsea_df['nes'])].copy()

print(f"      Loaded {len(gsea_df)} pathway enrichments")
print(f"      Cell types: {gsea_df['celltype'].nunique()}")
print(f"      Pathways: {gsea_df['pathway'].nunique()}")
print(f"      NES range: [{gsea_df['nes'].min():.2f}, {gsea_df['nes'].max():.2f}]")

# ============================================================================
# STEP 2: Parse gene lists from leading_edge column (WITH CHECKPOINT)
# ============================================================================
print("\n[2/6] Parsing gene lists from pathways...")

import pickle

if os.path.exists(CHECKPOINT_PATHWAY_GENES):
    print("      ✓ Loading cached pathway genes...")
    with open(CHECKPOINT_PATHWAY_GENES, 'rb') as f:
        pathway_genes = pickle.load(f)
    print(f"      Loaded {len(pathway_genes)} unique pathways")
else:
    print("      Computing pathway genes...")
    pathway_genes = {}
    for idx, row in gsea_df.iterrows():
        pathway_name = row['pathway']
        genes_str = row['leading_edge']
        
        if pd.notna(genes_str) and genes_str:
            genes = [g.strip() for g in genes_str.split(',')]
            
            if pathway_name not in pathway_genes:
                pathway_genes[pathway_name] = set()
            pathway_genes[pathway_name].update(genes)
    
    # Convert sets to lists
    for pathway in pathway_genes:
        pathway_genes[pathway] = list(pathway_genes[pathway])
    
    # Save checkpoint
    with open(CHECKPOINT_PATHWAY_GENES, 'wb') as f:
        pickle.dump(pathway_genes, f)
    print(f"      ✓ Saved checkpoint: {CHECKPOINT_PATHWAY_GENES}")

print(f"      Parsed {len(pathway_genes)} unique pathways")
print(f"      Total unique genes: {len(set([g for genes in pathway_genes.values() for g in genes]))}")

# ============================================================================
# STEP 3: Create gene-pathway co-occurrence matrix (WITH CHECKPOINT)
# ============================================================================
print("\n[3/6] Creating gene-pathway matrix...")

from scipy.sparse import save_npz, load_npz

if os.path.exists(CHECKPOINT_GENE_PATHWAY_MATRIX):
    print("      ✓ Loading cached gene-pathway matrix...")
    gene_pathway_matrix = load_npz(CHECKPOINT_GENE_PATHWAY_MATRIX).toarray()
    
    # Load gene and pathway lists
    with open(CHECKPOINT_PATHWAY_GENES.replace('.pkl', '_genes.pkl'), 'rb') as f:
        all_genes = pickle.load(f)
    with open(CHECKPOINT_PATHWAY_GENES.replace('.pkl', '_pathways.pkl'), 'rb') as f:
        all_pathways = pickle.load(f)
    
    print(f"      Matrix size: {len(all_genes)} genes × {len(all_pathways)} pathways")
else:
    print("      Computing gene-pathway matrix...")
    all_genes = sorted(set([g for genes in pathway_genes.values() for g in genes]))
    all_pathways = sorted(pathway_genes.keys())
    
    print(f"      Matrix size: {len(all_genes)} genes × {len(all_pathways)} pathways")
    
    gene_pathway_matrix = np.zeros((len(all_genes), len(all_pathways)))
    
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    pathway_to_idx = {pathway: idx for idx, pathway in enumerate(all_pathways)}
    
    for pathway, genes in pathway_genes.items():
        pathway_idx = pathway_to_idx[pathway]
        for gene in genes:
            gene_idx = gene_to_idx[gene]
            gene_pathway_matrix[gene_idx, pathway_idx] = 1
    
    # Save checkpoint
    save_npz(CHECKPOINT_GENE_PATHWAY_MATRIX, csr_matrix(gene_pathway_matrix))
    with open(CHECKPOINT_PATHWAY_GENES.replace('.pkl', '_genes.pkl'), 'wb') as f:
        pickle.dump(all_genes, f)
    with open(CHECKPOINT_PATHWAY_GENES.replace('.pkl', '_pathways.pkl'), 'wb') as f:
        pickle.dump(all_pathways, f)
    print(f"      ✓ Saved checkpoint: {CHECKPOINT_GENE_PATHWAY_MATRIX}")

print(f"      Matrix density: {(gene_pathway_matrix.sum() / gene_pathway_matrix.size) * 100:.2f}% non-zero")
print(f"      Average pathways per gene: {gene_pathway_matrix.sum(axis=1).mean():.1f}")

# ============================================================================
# STEP 4: Create gene embeddings using SVD (WITH CHECKPOINT)
# ============================================================================
print("\n[4/6] Creating gene embeddings using TruncatedSVD...")

if os.path.exists(OUTPUT_GENE_EMBEDDINGS):
    print("      ✓ Loading cached gene embeddings...")
    gene_embeddings_df = pd.read_csv(OUTPUT_GENE_EMBEDDINGS, index_col=0)
    n_components = len([c for c in gene_embeddings_df.columns if c.startswith('emb_')])
    print(f"      Gene embeddings shape: {gene_embeddings_df.shape}")
else:
    print("      Computing gene embeddings...")
    n_components = 128
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    gene_embeddings_matrix = svd.fit_transform(gene_pathway_matrix)
    
    print(f"      Gene embeddings shape: {gene_embeddings_matrix.shape}")
    print(f"      Explained variance: {svd.explained_variance_ratio_.sum() * 100:.1f}%")
    
    gene_embeddings_df = pd.DataFrame(
        gene_embeddings_matrix,
        index=all_genes,
        columns=[f'emb_{i}' for i in range(n_components)]
    )
    
    gene_embeddings_df.to_csv(OUTPUT_GENE_EMBEDDINGS)
    print(f"      ✓ Saved gene embeddings to {OUTPUT_GENE_EMBEDDINGS}")

# Visualize gene embeddings
if not os.path.exists(FIG_GENE_EMBEDDING_PCA):
    print("      Creating gene embedding visualization...")
    pca = PCA(n_components=2)
    gene_pca = pca.fit_transform(gene_embeddings_df.values)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(gene_pca[:, 0], gene_pca[:, 1], alpha=0.3, s=1)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('Gene Embeddings (PCA Visualization)')
    plt.tight_layout()
    plt.savefig(FIG_GENE_EMBEDDING_PCA, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved figure to {FIG_GENE_EMBEDDING_PCA}")
else:
    print(f"      ✓ Figure already exists: {FIG_GENE_EMBEDDING_PCA}")

# ============================================================================
# STEP 5: Load patient expression data (WITH CHECKPOINT)
# ============================================================================
print("\n[5/6] Loading patient expression data...")

adata = sc.read_h5ad(P2_H5AD_FILE)
print(f"      Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"      Donors: {adata.obs['donor_id'].nunique()}")
print(f"      Cell types: {adata.obs['annotation2v2'].nunique()}")

# Get common genes (WITH CHECKPOINT)
if os.path.exists(CHECKPOINT_COMMON_GENES):
    print("      ✓ Loading cached common genes...")
    with open(CHECKPOINT_COMMON_GENES, 'rb') as f:
        common_genes = pickle.load(f)
else:
    print("      Computing common genes...")
    adata_genes = adata.var_names.tolist()
    embedding_genes = gene_embeddings_df.index.tolist()
    common_genes = list(set(adata_genes) & set(embedding_genes))
    
    with open(CHECKPOINT_COMMON_GENES, 'wb') as f:
        pickle.dump(common_genes, f)
    print(f"      ✓ Saved checkpoint: {CHECKPOINT_COMMON_GENES}")

print(f"\n      Genes in adata: {adata.shape[1]}")
print(f"      Genes with embeddings: {len(gene_embeddings_df)}")
print(f"      Common genes: {len(common_genes)} ({len(common_genes)/adata.shape[1]*100:.1f}% coverage)")

adata_filtered = adata[:, common_genes].copy()
gene_embeddings_filtered = gene_embeddings_df.loc[common_genes]

# ============================================================================
# STEP 6: Create patient embeddings (WITH CHECKPOINT)
# ============================================================================
print("\n[6/6] Creating patient embeddings...")

if os.path.exists(OUTPUT_EMBEDDINGS):
    print("      ✓ Loading cached patient embeddings...")
    patient_embeddings_df = pd.read_csv(OUTPUT_EMBEDDINGS)
    print(f"      Loaded {len(patient_embeddings_df)} patient-celltype combinations")
else:
    print("      Computing patient embeddings...")
    patient_embeddings_list = []
    
    patient_celltypes = adata_filtered.obs[['donor_id', 'annotation2v2']].drop_duplicates()
    
    print(f"      Processing {len(patient_celltypes)} patient-celltype combinations...")
    
    for idx, (donor_id, celltype) in enumerate(patient_celltypes.values):
        if (idx + 1) % 50 == 0:
            print(f"      Progress: {idx+1}/{len(patient_celltypes)}")
        
        mask = (adata_filtered.obs['donor_id'] == donor_id) & \
               (adata_filtered.obs['annotation2v2'] == celltype)
        cells = adata_filtered[mask]
        
        if len(cells) == 0:
            continue
        
        if hasattr(cells.X, 'toarray'):
            mean_expr = cells.X.toarray().mean(axis=0)
        else:
            mean_expr = cells.X.mean(axis=0)
        
        weighted_embeddings = mean_expr[:, np.newaxis] * gene_embeddings_filtered.values
        patient_embedding = weighted_embeddings.sum(axis=0) / (mean_expr.sum() + 1e-10)
        
        result = {
            'donor_id': donor_id,
            'celltype': celltype,
            'n_cells': len(cells)
        }
        for i, val in enumerate(patient_embedding):
            result[f'emb_{i}'] = val
        
        patient_embeddings_list.append(result)
    
    patient_embeddings_df = pd.DataFrame(patient_embeddings_list)
    
    print(f"\n      Created embeddings for {len(patient_embeddings_df)} patient-celltype combinations")
    
    patient_embeddings_df.to_csv(OUTPUT_EMBEDDINGS, index=False)
    print(f"      ✓ Saved to {OUTPUT_EMBEDDINGS}")

# ============================================================================
# VISUALIZATION 1: Patient clustering
# ============================================================================
if not os.path.exists(FIG_PATIENT_CLUSTERING):
    print("\n[VIZ 1/3] Creating patient clustering visualization...")
    
    embedding_cols = [col for col in patient_embeddings_df.columns if col.startswith('emb_')]
    X = patient_embeddings_df[embedding_cols].values
    
    pca = PCA(n_components=2)
    patient_pca = pca.fit_transform(X)
    
    unique_celltypes = patient_embeddings_df['celltype'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_celltypes)))
    
    plt.figure(figsize=(12, 8))
    for i, celltype in enumerate(unique_celltypes):
        mask = patient_embeddings_df['celltype'] == celltype
        plt.scatter(patient_pca[mask, 0], patient_pca[mask, 1], 
                    label=celltype if len(celltype) < 30 else celltype[:27]+'...',
                    alpha=0.6, s=50, color=colors[i])
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('Patient Embeddings Colored by Cell Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_PATIENT_CLUSTERING, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved to {FIG_PATIENT_CLUSTERING}")
else:
    print(f"\n[VIZ 1/3] ✓ Figure already exists: {FIG_PATIENT_CLUSTERING}")

# ============================================================================
# VISUALIZATION 2: Embedding distribution
# ============================================================================
if not os.path.exists(FIG_EMBEDDING_DISTRIBUTION):
    print("\n[VIZ 2/3] Creating embedding distribution visualization...")
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(10):
        emb_col = f'emb_{i}'
        axes[i].hist(patient_embeddings_df[emb_col], bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Dimension {i}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
    
    plt.suptitle('Distribution of First 10 Embedding Dimensions')
    plt.tight_layout()
    plt.savefig(FIG_EMBEDDING_DISTRIBUTION, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Saved to {FIG_EMBEDDING_DISTRIBUTION}")
else:
    print(f"\n[VIZ 2/3] ✓ Figure already exists: {FIG_EMBEDDING_DISTRIBUTION}")

# ============================================================================
# VIZUALIZATION 3: Calculate pathway activity scores (WITH CHECKPOINT)
# ============================================================================
print("\n[VIZ 3/3] Calculating pathway activity scores...")

if os.path.exists(OUTPUT_PATHWAY_SCORES):
    print(f"      ✓ Pathway scores already exist at {OUTPUT_PATHWAY_SCORES}")
    pathway_scores_df = pd.read_csv(OUTPUT_PATHWAY_SCORES)
    print(f"      Loaded {len(pathway_scores_df)} pathway scores")
else:
    print("      Computing pathway scores (this may take a few minutes)...")
    pathway_scores_list = []
    
    for idx, row in patient_embeddings_df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"      Progress: {idx+1}/{len(patient_embeddings_df)}")
        
        donor_id = row['donor_id']
        celltype = row['celltype']
        
        mask = (adata_filtered.obs['donor_id'] == donor_id) & \
               (adata_filtered.obs['annotation2v2'] == celltype)
        cells = adata_filtered[mask]
        
        if len(cells) == 0:
            continue
        
        if hasattr(cells.X, 'toarray'):
            mean_expr = cells.X.toarray().mean(axis=0)
        else:
            mean_expr = cells.X.mean(axis=0)
        
        for pathway, genes in pathway_genes.items():
            pathway_genes_in_data = [g for g in genes if g in common_genes]
            
            if len(pathway_genes_in_data) == 0:
                continue
            
            gene_indices = [common_genes.index(g) for g in pathway_genes_in_data]
            pathway_score = mean_expr[gene_indices].mean()
            
            pathway_scores_list.append({
                'donor_id': donor_id,
                'celltype': celltype,
                'pathway': pathway,
                'pathway_score': pathway_score,
                'n_genes': len(pathway_genes_in_data)
            })
    
    pathway_scores_df = pd.DataFrame(pathway_scores_list)
    pathway_scores_df.to_csv(OUTPUT_PATHWAY_SCORES, index=False)
    print(f"      ✓ Saved {len(pathway_scores_df)} pathway scores to {OUTPUT_PATHWAY_SCORES}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("P3 COMPLETE - COLON!")
print("="*80)

print("\nOUTPUTS GENERATED:")
print(f"   ✓ Patient embeddings: {OUTPUT_EMBEDDINGS}")
print(f"     - {len(patient_embeddings_df)} patient-celltype combinations")
print(f"     - {n_components} dimensions per patient")

print(f"\n   ✓ Gene embeddings: {OUTPUT_GENE_EMBEDDINGS}")
print(f"     - {len(gene_embeddings_df)} genes")
print(f"     - {n_components} dimensions per gene")

print(f"\n   ✓ Pathway scores: {OUTPUT_PATHWAY_SCORES}")
print(f"     - {len(pathway_scores_df)} pathway-patient combinations")

print("\nFIGURES GENERATED:")
print(f"   ✓ Patient clustering: {FIG_PATIENT_CLUSTERING}")
print(f"   ✓ Embedding distribution: {FIG_EMBEDDING_DISTRIBUTION}")
print(f"   ✓ Gene embedding PCA: {FIG_GENE_EMBEDDING_PCA}")

print("\nNext step: Run P4_colon.py to match patients to drugs!")
print("="*80)