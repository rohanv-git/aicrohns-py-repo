"""
P3: Pathway Embeddings via SVD - Small Intestine
Works with P2's actual output: condition-level GSEA (not patient-level)
Creates pathway embeddings per condition based on cell-type enrichment patterns
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_palette("husl")
FIG_DIR = Path("figures/P3_Embeddings")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_DIR = Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

P2_FILE = CHECKPOINT_DIR / 'P2_gsea_results_filtered.pkl'
OUTPUT_FILE = CHECKPOINT_DIR / 'P3_pathway_embeddings.pkl'

N_COMPONENTS = 20
FDR_THRESHOLD = 0.25

print("="*80)
print("P3: PATHWAY EMBEDDINGS VIA SVD - SMALL INTESTINE")
print("REVISED: Works with condition-level GSEA from P2")
print("="*80)

# ============================================================================
# [1/4] Load P2 GSEA Results
# ============================================================================
print("[1/4] Loading P2 GSEA results...")
gsea_results = pd.read_pickle(P2_FILE)

print(f"      Loaded {len(gsea_results)} enrichments")
print(f"      Columns: {list(gsea_results.columns)}")
print(f"      Unique pathways: {gsea_results['pathway'].nunique()}")
print(f"      Unique cell types: {gsea_results['cell_type'].nunique()}")

# Check if condition column exists
if 'condition' in gsea_results.columns:
    print(f"      Conditions: {gsea_results['condition'].unique()}")
else:
    print("      WARNING: No 'condition' column found")
    print("      Sample data:")
    print(gsea_results.head())

# VISUALIZATION 1: P2 Data Overview
print("\\n  📊 Visualizing P2 input data...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Enrichments per cell type
ct_counts = gsea_results.groupby('cell_type').size().sort_values(ascending=False).head(15)
axes[0].barh(range(len(ct_counts)), ct_counts.values, color='#667eea')
axes[0].set_yticks(range(len(ct_counts)))
axes[0].set_yticklabels([ct[:30] for ct in ct_counts.index], fontsize=8)
axes[0].set_xlabel('Number of Enrichments', fontweight='bold')
axes[0].set_title('A) Input: Enrichments per Cell Type', fontweight='bold')
axes[0].invert_yaxis()

# Panel B: Pathway frequency
pathway_counts = gsea_results.groupby('pathway').size().sort_values(ascending=False).head(15)
axes[1].barh(range(len(pathway_counts)), pathway_counts.values, color='#764ba2')
axes[1].set_yticks(range(len(pathway_counts)))
axes[1].set_yticklabels([f"Pathway {p}" for p in pathway_counts.index], fontsize=8)
axes[1].set_xlabel('Frequency', fontweight='bold')
axes[1].set_title('B) Input: Top Pathways', fontweight='bold')
axes[1].invert_yaxis()

# Panel C: NES distribution
axes[2].hist(gsea_results['nes'], bins=50, color='#2ECC71', alpha=0.7, edgecolor='black')
axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[2].set_xlabel('NES', fontweight='bold')
axes[2].set_ylabel('Frequency', fontweight='bold')
axes[2].set_title('C) Input: NES Distribution', fontweight='bold')

plt.suptitle('P3 Input Data from P2 - Small Intestine', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "Step1_Input_Data.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: Step1_Input_Data.png")
# ============================================================================
# [2/4] Filter to Significant Pathways
# ============================================================================
print(f"\n[2/4] Filtering pathways (FDR < {FDR_THRESHOLD})...")
significant = gsea_results[gsea_results['fdr'] < FDR_THRESHOLD].copy()
significant_pathways = sorted(significant['pathway'].unique())
cell_types = sorted(significant['cell_type'].unique())

print(f"      Significant pathways: {len(significant_pathways)}")
print(f"      Cell types: {len(cell_types)}")

# VISUALIZATION 2: Pathway Filtering
print("\\n  📊 Visualizing pathway filtering...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Before/After filtering
axes[0, 0].bar(['All\\nPathways', f'Significant\\n(FDR<{FDR_THRESHOLD})'],
               [gsea_results['pathway'].nunique(), len(significant_pathways)],
               color=['#95A5A6', '#2ECC71'], alpha=0.8, edgecolor='black')
axes[0, 0].set_ylabel('Number of Pathways', fontweight='bold')
axes[0, 0].set_title('A) Filtering Effect', fontweight='bold')
for i, v in enumerate([gsea_results['pathway'].nunique(), len(significant_pathways)]):
    axes[0, 0].text(i, v + max(gsea_results['pathway'].nunique(), len(significant_pathways))*0.02,
                   f'{v}\\n({v/gsea_results["pathway"].nunique()*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')

# Panel B: FDR distribution
axes[0, 1].hist(np.log10(gsea_results['fdr'] + 1e-10), bins=50, color='#667eea', 
               alpha=0.7, edgecolor='black', label='All')
axes[0, 1].hist(np.log10(significant['fdr'] + 1e-10), bins=50, color='#2ECC71',
               alpha=0.7, edgecolor='black', label='Significant')
axes[0, 1].axvline(x=np.log10(FDR_THRESHOLD), color='red', linestyle='--', 
                  linewidth=2, label=f'FDR = {FDR_THRESHOLD}')
axes[0, 1].set_xlabel('log10(FDR)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('B) FDR Distribution', fontweight='bold')
axes[0, 1].legend()

# Panel C: Cell type coverage
axes[1, 0].bar(['Input\\nCell Types', 'Output\\nCell Types'],
               [gsea_results['cell_type'].nunique(), len(cell_types)],
               color=['#667eea', '#764ba2'], alpha=0.8, edgecolor='black')
axes[1, 0].set_ylabel('Number of Cell Types', fontweight='bold')
axes[1, 0].set_title('C) Cell Type Coverage', fontweight='bold')
for i, v in enumerate([gsea_results['cell_type'].nunique(), len(cell_types)]):
    axes[1, 0].text(i, v + max(gsea_results['cell_type'].nunique(), len(cell_types))*0.02,
                   f'{v}', ha='center', va='bottom', fontweight='bold')

# Panel D: Summary table
summary_data = [
    ['Total Pathways (Input)', gsea_results['pathway'].nunique()],
    ['Significant Pathways', len(significant_pathways)],
    ['Filtering Rate', f"{len(significant_pathways)/gsea_results['pathway'].nunique()*100:.1f}%"],
    ['Cell Types', len(cell_types)],
    ['Embedding Dimension', N_COMPONENTS]
]
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=summary_data, cellLoc='left', loc='center',
                        bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
for i in range(len(summary_data)):
    table[(i, 0)].set_facecolor('#667eea')
    table[(i, 0)].set_text_props(weight='bold', color='white')
axes[1, 1].set_title('D) Summary Statistics', fontweight='bold', pad=20)

plt.suptitle('P3 Pathway Filtering - Small Intestine', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "Step2_Pathway_Filtering.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: Step2_Pathway_Filtering.png")
# ============================================================================
# [3/4] Build Pathway × Cell Type Matrix & Perform SVD
# ============================================================================
print("\n[3/4] Building pathway embeddings via SVD...")

# Separate by condition if available
if 'condition' in significant.columns:
    healthy_gsea = significant[significant['condition'] == 'Healthy']
    crohns_gsea = significant[significant['condition'] == "Crohn's"]
    has_conditions = True
else:
    # If no condition column, treat all as single group
    healthy_gsea = significant
    crohns_gsea = pd.DataFrame()
    has_conditions = False

def build_pathway_embeddings(gsea_df, pathways, cell_types, n_components):
    """
    Build (n_pathways, n_components) embeddings
    Each pathway gets a low-dimensional representation based on its enrichment pattern across cell types
    """
    n_pathways = len(pathways)
    n_cell_types = len(cell_types)
    
    # Build pathway × cell_type matrix of NES scores
    matrix = np.zeros((n_pathways, n_cell_types))
    
    for p_idx, pathway in enumerate(pathways):
        pathway_data = gsea_df[gsea_df['pathway'] == pathway]
        
        for _, row in pathway_data.iterrows():
            try:
                celltype_idx = cell_types.index(row['cell_type'])
                matrix[p_idx, celltype_idx] = row['nes']
            except ValueError:
                continue
    
    # SVD on pathway × cell_type matrix
    if matrix.any():
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        # Pathway embeddings: first n_components from U weighted by singular values
        n_comp = min(n_components, len(S))
        embeddings = np.zeros((n_pathways, n_components))
        embeddings[:, :n_comp] = U[:, :n_comp] * S[:n_comp]
        return embeddings, matrix
    else:
        return np.zeros((n_pathways, n_components)), matrix

if has_conditions:
    print("      Computing healthy embeddings...")
    healthy_embeddings, healthy_matrix = build_pathway_embeddings(
        healthy_gsea, significant_pathways, cell_types, N_COMPONENTS
    )
    
    print("      Computing Crohn's embeddings...")
    crohns_embeddings, crohns_matrix = build_pathway_embeddings(
        crohns_gsea, significant_pathways, cell_types, N_COMPONENTS
    )
    
    print(f"      Healthy embeddings: {healthy_embeddings.shape}")
    print(f"      Crohn's embeddings: {crohns_embeddings.shape}")
else:
    print("      Computing embeddings (no condition separation)...")
    embeddings, matrix = build_pathway_embeddings(
        healthy_gsea, significant_pathways, cell_types, N_COMPONENTS
    )
    healthy_embeddings = embeddings
    crohns_embeddings = embeddings
    print(f"      Embeddings: {embeddings.shape}")
# VISUALIZATION 3: Embedding Analysis
print("\\n  📊 Visualizing pathway embeddings...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Panel A: Embedding norms
ax1 = fig.add_subplot(gs[0, 0])
norms = np.linalg.norm(healthy_embeddings, axis=1)
ax1.hist(norms, bins=30, color='#667eea', alpha=0.7, edgecolor='black')
ax1.axvline(norms.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {norms.mean():.2f}')
ax1.set_xlabel('Embedding Norm', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('A) Embedding Magnitude Distribution', fontweight='bold')
ax1.legend()

# Panel B: Variance explained (if possible)
ax2 = fig.add_subplot(gs[0, 1])
try:
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(healthy_embeddings)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum)+1), cumsum, 'b-', linewidth=2, label='Cumulative')
    ax2.plot(range(1, len(pca.explained_variance_ratio_)+1),
            pca.explained_variance_ratio_, 'r--', linewidth=2, alpha=0.7, label='Individual')
    ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95%')
    ax2.set_xlabel('Component', fontweight='bold')
    ax2.set_ylabel('Variance Explained', fontweight='bold')
    ax2.set_title('B) Variance Explained', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
except:
    ax2.text(0.5, 0.5, 'PCA analysis unavailable', ha='center', va='center',
            transform=ax2.transAxes)

# Panel C: Component loadings heatmap
ax3 = fig.add_subplot(gs[0, 2])
n_show = min(30, healthy_embeddings.shape[0])
loadings = healthy_embeddings[:n_show, :]
im = ax3.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
ax3.set_yticks(range(0, n_show, 5))
ax3.set_yticklabels([f'P{i}' for i in range(0, n_show, 5)])
ax3.set_xlabel('Component', fontweight='bold')
ax3.set_ylabel('Pathway', fontweight='bold')
ax3.set_title('C) Component Loadings (Top 30)', fontweight='bold')
plt.colorbar(im, ax=ax3, label='Loading')

# Panel D: Embedding shape info
ax4 = fig.add_subplot(gs[1, 0])
ax4.axis('off')
shape_data = [
    ['Pathways', healthy_embeddings.shape[0]],
    ['Components', healthy_embeddings.shape[1]],
    ['Total Parameters', healthy_embeddings.shape[0] * healthy_embeddings.shape[1]]
]
table = ax4.table(cellText=shape_data, cellLoc='center', loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)
for i in range(len(shape_data)):
    table[(i, 0)].set_facecolor('#667eea')
    table[(i, 0)].set_text_props(weight='bold', color='white')
ax4.set_title('D) Embedding Dimensions', fontweight='bold', pad=20)

# Panel E: Per-component variance
ax5 = fig.add_subplot(gs[1, 1:])
component_var = np.var(healthy_embeddings, axis=0)
ax5.bar(range(len(component_var)), component_var, color='#764ba2', alpha=0.8, edgecolor='black')
ax5.set_xlabel('Component Index', fontweight='bold')
ax5.set_ylabel('Variance', fontweight='bold')
ax5.set_title('E) Variance per Component', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# Panel F: Top pathways by norm
ax6 = fig.add_subplot(gs[2, :])
top_idx = np.argsort(norms)[-20:]
top_norms = norms[top_idx]
pathway_names_short = [f"Pathway {significant_pathways[i]}" for i in top_idx]
colors_top = plt.cm.plasma(np.interp(top_norms, [top_norms.min(), top_norms.max()], [0.2, 0.9]))
ax6.barh(range(len(top_idx)), top_norms, color=colors_top)
ax6.set_yticks(range(len(top_idx)))
ax6.set_yticklabels(pathway_names_short, fontsize=8)
ax6.set_xlabel('Embedding Norm', fontweight='bold')
ax6.set_title('F) Top 20 Pathways by Embedding Magnitude', fontweight='bold')
ax6.invert_yaxis()

plt.suptitle('P3 Pathway Embedding Analysis - Small Intestine', fontsize=16, fontweight='bold')
plt.savefig(f'figures/P3_Embeddings/Step3_Embedding_Analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: Step3_Embedding_Analysis.png")
# ============================================================================
# [4/4] Create Filtered pathway_lookup and Save
# ============================================================================
print("\n[4/4] Creating filtered pathway_lookup...")

# Create minimal pathway lookup (can be enhanced with full pathway DB if available)
filtered_pathway_lookup = {
    pathway: {'name': pathway} for pathway in significant_pathways
}

# Try to load full pathway database for richer metadata
try:
    with open(CHECKPOINT_DIR / 'P2_pathway_database.pkl', 'rb') as f:
        full_pathway_db = pickle.load(f)
    filtered_pathway_lookup = {
        pathway: full_pathway_db.get(pathway, {'name': pathway})
        for pathway in significant_pathways
    }
    print(f"      Loaded pathway metadata from P2_pathway_database.pkl")
except FileNotFoundError:
    print(f"      Using minimal pathway lookup (only names)")

# Verification checks
assert len(filtered_pathway_lookup) == len(significant_pathways), \
    f"Lookup size mismatch: {len(filtered_pathway_lookup)} != {len(significant_pathways)}"
assert len(filtered_pathway_lookup) == healthy_embeddings.shape[0], \
    f"Lookup-embedding mismatch: {len(filtered_pathway_lookup)} != {healthy_embeddings.shape[0]}"
assert set(filtered_pathway_lookup.keys()) == set(significant_pathways), \
    "Lookup keys don't match significant pathways"

print(f"      ✓ Filtered pathway_lookup: {len(filtered_pathway_lookup)} pathways")
print(f"      ✓ Matches embedding dimension: {healthy_embeddings.shape[0]}")

# Save with filtered lookup
output_data = {
    'healthy_embeddings': healthy_embeddings,      # (n_pathways, n_components)
    'crohns_embeddings': crohns_embeddings,        # (n_pathways, n_components)
    'pathway_lookup': filtered_pathway_lookup,     # Only significant pathways
    'pathway_order': significant_pathways,         # Explicit ordering
    'cell_types': cell_types,
    'n_components': N_COMPONENTS,
    'has_conditions': has_conditions
}

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(output_data, f)

print(f"      ✓ Saved to {OUTPUT_FILE}")
# VISUALIZATION 4: Final Summary
print("\\n  📊 Creating final summary visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Input vs Output
axes[0, 0].bar(['Input\\nEnrichments', 'Output\\nPathways', 'Output\\nComponents'],
               [len(gsea_results), len(significant_pathways), N_COMPONENTS],
               color=['#95A5A6', '#667eea', '#764ba2'], alpha=0.8, edgecolor='black')
axes[0, 0].set_ylabel('Count', fontweight='bold')
axes[0, 0].set_title('A) Pipeline Transformation', fontweight='bold')
for i, v in enumerate([len(gsea_results), len(significant_pathways), N_COMPONENTS]):
    axes[0, 0].text(i, v + max([len(gsea_results), len(significant_pathways), N_COMPONENTS])*0.02,
                   f'{v}', ha='center', va='bottom', fontweight='bold')

# Panel B: Embedding statistics
axes[0, 1].axis('off')
stats_data = [
    ['Metric', 'Value'],
    ['Pathways Embedded', str(healthy_embeddings.shape[0])],
    ['Embedding Dimension', str(N_COMPONENTS)],
    ['Mean Norm', f'{norms.mean():.3f}'],
    ['Std Norm', f'{norms.std():.3f}'],
    ['Min Norm', f'{norms.min():.3f}'],
    ['Max Norm', f'{norms.max():.3f}']
]
table = axes[0, 1].table(cellText=stats_data, cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
table[(0, 0)].set_facecolor('#667eea')
table[(0, 1)].set_facecolor('#667eea')
table[(0, 0)].set_text_props(weight='bold', color='white')
table[(0, 1)].set_text_props(weight='bold', color='white')
axes[0, 1].set_title('B) Embedding Statistics', fontweight='bold', pad=20)

# Panel C: Dimensionality reduction (UMAP if available)
ax = axes[1, 0]
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(healthy_embeddings)
    ax.scatter(coords[:, 0], coords[:, 1], c=norms, cmap='viridis', s=50, alpha=0.6, edgecolors='black')
    ax.set_xlabel('UMAP 1', fontweight='bold')
    ax.set_ylabel('UMAP 2', fontweight='bold')
    ax.set_title('C) UMAP Projection', fontweight='bold')
    plt.colorbar(ax.collections[0], ax=ax, label='Norm')
except:
    ax.text(0.5, 0.5, 'UMAP unavailable\\n(install with: pip install umap-learn)',
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('C) UMAP Projection (Not Available)', fontweight='bold')

# Panel D: Cell type coverage
axes[1, 1].bar(range(min(15, len(cell_types))), 
              [len(significant[significant['cell_type'] == ct]) for ct in cell_types[:15]],
              color=plt.cm.tab20(np.linspace(0, 1, min(15, len(cell_types)))))
axes[1, 1].set_xticks(range(min(15, len(cell_types))))
axes[1, 1].set_xticklabels([ct[:10] for ct in cell_types[:15]], rotation=45, ha='right', fontsize=8)
axes[1, 1].set_ylabel('Enrichments', fontweight='bold')
axes[1, 1].set_title('D) Cell Type Coverage (Top 15)', fontweight='bold')

plt.suptitle('P3 Final Summary - Small Intestine', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'figures/P3_Embeddings/Step4_Final_Summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: Step4_Final_Summary.png")
print("\n" + "="*80)
print("P3 COMPLETE - SUMMARY")
print("="*80)
print(f"Pathways: {len(significant_pathways)}")
print(f"Cell types: {len(cell_types)}")
print(f"Embeddings: {N_COMPONENTS}D per pathway")
print(f"pathway_lookup: {len(filtered_pathway_lookup)} pathways (FILTERED ✓)")
print(f"Structure: condition-level (not patient-level)")
print("="*80)
print("\nNOTE: P2 provides condition-level GSEA (Healthy vs Crohn's aggregate),")
print("not patient-specific results. P3 creates pathway embeddings based on")
print("cell-type enrichment patterns, which P4 can then use for predictions.")
print("="*80)