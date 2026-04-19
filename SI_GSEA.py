"""
P2: Population-Level Drug2Cell Analysis - SMALL INTESTINE
WITH HEALTHY CONTROLS in addition to Crohn's patients
"""

import os
import pickle
import scanpy as sc
import pandas as pd
import os
import numpy as np
from blitzgsea import enrichr
import gseapy as gp
import drug2cell as d2c
from scipy.stats import hypergeom
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIG_DIR = Path("figures/P2_GSEA")
FIG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = Path("checkpoints")
FIG_DIR = Path("figures/P2_GSEA")

CHECKPOINT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_palette("husl")

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/rvellamcheti/Downloads/Cleaned_raw_annotated_object_LK.v2.h5ad'
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Parameters
NES_THRESHOLD = 1.5
FDR_THRESHOLD = 0.25
TISSUE = 'small intestine'

# Map disease codes to readable labels
disease_map = {
    'MONDO_0005011': "Crohn's",
    'PATO_0000461': 'Healthy'
}


print("=" * 80)
print("P2: POPULATION-LEVEL DRUG2CELL ANALYSIS - SMALL INTESTINE")
print("WITH HEALTHY CONTROLS + MSigDB PATHWAYS")
print("=" * 80)
print(f"Filtering: |NES| > {NES_THRESHOLD}, FDR < {FDR_THRESHOLD}")
print()

# ============================================================================
# STEP 1: Load and filter data
# ============================================================================
print("[1/7] Loading and filtering small intestine data...")
checkpoint_path = f'{CHECKPOINT_DIR}/P2_filtered.h5ad'

if os.path.exists(checkpoint_path):
    adata = sc.read_h5ad(checkpoint_path)
    print(f"      Loaded: {adata.shape}")
else:
    # Load full dataset
    adata = sc.read_h5ad(DATA_PATH)
    print(f"      Total cells before filtering: {adata.n_obs}")
    
    # Filter for small intestine
    adata = adata[adata.obs['tissue'] == TISSUE].copy()
    print(f"      After tissue filtering (small intestine): {adata.n_obs} cells")
    
    # Check disease status
    print(f"      Available disease labels: {adata.obs['disease'].cat.categories.tolist()}")
    print(adata.obs['disease'].cat.categories)
    
    # MONDO_0005011 = Crohn's disease
    # PATO_0000461 = healthy/normal
    healthy_label = 'PATO_0000461'
    crohns_label = 'MONDO_0005011'
    
    print(f"      Found healthy controls labeled as: '{healthy_label}'")
    
    # Keep both healthy and Crohn's
    adata = adata[adata.obs['disease'].isin([healthy_label, crohns_label])].copy()
    print(f"      After disease filtering: {adata.n_obs} cells")
    
    # Show breakdown
    print(f"      Disease breakdown:")
    print(adata.obs['disease'].value_counts())
    print(f"      Patient breakdown by disease:")
    print(adata.obs.groupby('disease')['donor_id'].nunique())
    
    # Add disease status column for clarity
    adata.obs['disease_status'] = adata.obs['disease']
    
    # Save checkpoint
    adata.write_h5ad(checkpoint_path)
    print(f"      Saved: {adata.shape}")

print()
# VISUALIZATION 1: Data Filtering Summary
print("\n  📊 Generating filtering visualization...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Cells by tissue
tissue_counts = adata.obs['tissue'].value_counts().head(10)
axes[0].barh(range(len(tissue_counts)), tissue_counts.values, color='#667eea')
axes[0].set_yticks(range(len(tissue_counts)))
axes[0].set_yticklabels(tissue_counts.index, fontsize=9)
axes[0].set_xlabel('Number of Cells', fontweight='bold')
axes[0].set_title('A) Cells by Tissue', fontweight='bold')
axes[0].invert_yaxis()

# Panel B: Disease distribution
disease_counts = adata.obs['disease'].value_counts()
disease_labels = [disease_map.get(d, d) for d in disease_counts.index]
colors = ['#E74C3C' if 'Crohn' in str(l) else '#3498DB' for l in disease_labels]
axes[1].bar(range(len(disease_counts)), disease_counts.values, color=colors)
axes[1].set_xticks(range(len(disease_counts)))
axes[1].set_xticklabels(disease_labels, rotation=45, ha='right')
axes[1].set_ylabel('Number of Cells', fontweight='bold')
axes[1].set_title('B) Disease Distribution', fontweight='bold')

# Panel C: Patients by disease
patient_disease = adata.obs.groupby('donor_id')['disease'].first()
patient_counts = pd.Series([disease_map.get(d, d) for d in patient_disease]).value_counts()
axes[2].bar(range(len(patient_counts)), patient_counts.values, color=colors[:len(patient_counts)])
axes[2].set_xticks(range(len(patient_counts)))
axes[2].set_xticklabels(patient_counts.index, rotation=45, ha='right')
axes[2].set_ylabel('Number of Patients', fontweight='bold')
axes[2].set_title('C) Patients by Disease', fontweight='bold')

plt.suptitle(f'P2 Data Filtering Summary - {TISSUE.title()}', fontsize=14, fontweight='bold')
plt.tight_layout()
print("Figure dir exists:", FIG_DIR.exists())
plt.savefig(
    FIG_DIR / f"Step1_DataFiltering_{TISSUE.replace(' ', '')}.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close()
print("  ✓ Saved: Step1_DataFiltering.png")
# ============================================================================
# STEP 2: UMAP (if needed)
# ============================================================================
print("[2/7] UMAP exists, skipping...")
print()

# ============================================================================
# STEP 3: Drug2Cell scoring
# ============================================================================
print("[3/7] Running drug2cell scoring...")
checkpoint_path = f'{CHECKPOINT_DIR}/P2_drug2cell_scored.h5ad'

if os.path.exists(checkpoint_path):
    adata = sc.read_h5ad(checkpoint_path)
else:
    d2c.score(adata)
    adata.write_h5ad(checkpoint_path)

print("      ✓ Scored")
print()

# ============================================================================
# STEP 4: Rank genes
# ============================================================================
print("[4/7] Ranking genes...")
checkpoint_path = f'{CHECKPOINT_DIR}/P2_drug2cell_ranked.h5ad'

if os.path.exists(checkpoint_path):
    adata = sc.read_h5ad(checkpoint_path)
else:
    # Drug2cell stores scores in adata.uns['drug2cell'] as a separate AnnData object
    # But we need to rank GENES, not drugs, for pathway analysis
    # The drug2cell scores are drug × cell, but we need gene × cell for GSEA
    
    # Get the original gene expression data with cell type annotations
    drug2cell_adata = adata.uns['drug2cell']
    
    # Add cell type annotations to drug2cell object
    drug2cell_adata.obs['annotation2v2'] = adata.obs['annotation2v2']
    
    # Now rank the ORIGINAL GENES (not drugs) by their expression in each cell type
    # This gives us gene-level signatures for pathway analysis
    sc.tl.rank_genes_groups(
        adata,  # Use original adata with GENE expression
        groupby='annotation2v2',
        use_raw=False,
        method='wilcoxon'
    )
    
    adata.write_h5ad(checkpoint_path)

print("      ✓ Ranked")
print()
# VISUALIZATION 2: Gene Ranking Summary
print("\\n  📊 Generating gene ranking visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Number of genes per cell type
cell_types = adata.obs['annotation2v2'].cat.categories.tolist()
n_genes_per_ct = []
for ct in cell_types[:15]:  # Top 15
    try:
        genes = adata.uns['rank_genes_groups']['names'][ct]
        n_genes_per_ct.append(len([g for g in genes if g is not None]))
    except:
        n_genes_per_ct.append(0)

axes[0, 0].barh(range(min(15, len(cell_types))), n_genes_per_ct, color='#764ba2')
axes[0, 0].set_yticks(range(min(15, len(cell_types))))
axes[0, 0].set_yticklabels([ct[:30] for ct in cell_types[:15]], fontsize=8)
axes[0, 0].set_xlabel('Number of Ranked Genes', fontweight='bold')
axes[0, 0].set_title('A) Genes Ranked per Cell Type', fontweight='bold')
axes[0, 0].invert_yaxis()

# Panel B: Top gene scores distribution
if len(cell_types) > 0:
    ct = cell_types[0]
    try:
        scores = adata.uns['rank_genes_groups']['scores'][ct]
        scores_clean = [s for s in scores if s is not None and not np.isnan(s)]
        axes[0, 1].hist(scores_clean[:1000], bins=50, color='#667eea', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Ranking Score', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].set_title(f'B) Score Distribution ({ct[:30]})', fontweight='bold')
        axes[0, 1].axvline(np.median(scores_clean[:1000]), color='red', linestyle='--', label='Median')
        axes[0, 1].legend()
    except:
        axes[0, 1].text(0.5, 0.5, 'Score data unavailable', ha='center', va='center', 
                       transform=axes[0, 1].transAxes)

# Panel C: Cell type abundance
ct_counts = adata.obs['annotation2v2'].value_counts().head(15)
axes[1, 0].barh(range(len(ct_counts)), ct_counts.values, 
               color=plt.cm.viridis(np.linspace(0.2, 0.9, len(ct_counts))))
axes[1, 0].set_yticks(range(len(ct_counts)))
axes[1, 0].set_yticklabels([ct[:30] for ct in ct_counts.index], fontsize=8)
axes[1, 0].set_xlabel('Number of Cells', fontweight='bold')
axes[1, 0].set_title('C) Top 15 Cell Types by Abundance', fontweight='bold')
axes[1, 0].invert_yaxis()

# Panel D: Summary statistics table
ranked_cts = adata.uns['rank_genes_groups']['names'].dtype.names

summary_data = [
    ['Total Cell Types', len(cell_types)],
    ['Cells Analyzed', adata.n_obs],
    ['Genes per Cell', adata.n_vars],
    ['Ranked Cell Types', len(ranked_cts)]
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

plt.suptitle(f'P2 Gene Ranking Summary - {TISSUE.title()}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'figures/P2_GSEA/Step4_GeneRanking_{TISSUE.replace(" ", "")}.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: Step4_GeneRanking.png")
# ============================================================================
# STEP 5: Hypergeometric tests (optional, not used currently)
# ============================================================================
print("[5/7] Running hypergeometric tests...")

# Load MSigDB gene sets
print("      Loading MSigDB gene sets...")
gene_sets_dict = {}

# KEGG 2021 Human
try:
    print("        - KEGG_2021_Human...")
    kegg = gp.get_library(name='KEGG_2021_Human', organism='Human')
    for pathway, genes in kegg.items():
        gene_sets_dict[f"KEGG_{pathway}"] = genes
    print(f"          ✓ Loaded {len(kegg)} KEGG pathways")
except Exception as e:
    print(f"          ⚠ Warning: {e}")

# Hallmark gene sets
try:
    print("        - MSigDB_Hallmark_2020...")
    hallmark = gp.get_library(name='MSigDB_Hallmark_2020', organism='Human')
    for pathway, genes in hallmark.items():
        gene_sets_dict[f"Hallmark_{pathway}"] = genes
    print(f"          ✓ Loaded {len(hallmark)} Hallmark pathways")
except Exception as e:
    print(f"          ⚠ Warning: {e}")

# Reactome (Canonical)
try:
    print("        - Reactome_2022...")
    reactome = gp.get_library(name='Reactome_2022', organism='Human')
    for pathway, genes in reactome.items():
        gene_sets_dict[f"Reactome_{pathway}"] = genes
    print(f"          ✓ Loaded {len(reactome)} Reactome pathways")
except Exception as e:
    print(f"          ⚠ Warning: {e}")

print(f"      Total MSigDB gene sets: {len(gene_sets_dict)}")

# Hypergeometric test (placeholder - not actively used)
print(f"      ✓ 0 pathway-celltype associations tested")
print()

# ============================================================================
# STEP 6: GSEA using gseapy.prerank
# ============================================================================
print("[6/7] Running GSEA (30-60 minutes with 2186 pathways)...")
checkpoint_path = f'{CHECKPOINT_DIR}/P2_gsea_results.pkl'

if os.path.exists(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
        if isinstance(checkpoint_data, dict):
            gsea_df = checkpoint_data['gsea_df']
            pathway_lookup = checkpoint_data.get('pathway_lookup', {})
        else:
            gsea_df = checkpoint_data
            pathway_lookup = {}
else:
    gsea_results_list = []
    pathway_lookup = {}
    
    cell_types = adata.obs['annotation2v2'].cat.categories.tolist()
    
    for idx, celltype in enumerate(cell_types, 1):
        print(f"      [{idx}/{len(cell_types)}] {celltype}...")
        
        try:
            # Get ranked genes for this cell type
            rank_result = adata.uns['rank_genes_groups']
            genes = rank_result['names'][celltype]
            scores = rank_result['scores'][celltype]
            
            # Create ranking - genes should already be symbols now
            ranking = pd.Series(scores, index=genes).sort_values(ascending=False)
            
            # Run GSEA - FIXED: Pass gene_sets_dict instead of string
            try:
                gsea_res = gp.prerank(
                    rnk=ranking,
                    gene_sets=gene_sets_dict,  # ← FIXED: Use the dictionary variable
                    processes=4,
                    permutation_num=100,
                    seed=42,
                    outdir=None,
                    verbose=False
                )
                
                # Extract results
                res_df = gsea_res.res2d
                
                for i in range(len(res_df)):
                    nes = res_df.iloc[i]['NES']
                    pval = res_df.iloc[i]['NOM p-val']
                    fdr = res_df.iloc[i]['FDR q-val']
                    leading_edge = res_df.iloc[i].get('Lead_genes', '')
                    pathway_name = res_df.iloc[i].get('Term', f"Pathway_{i}")
                    
                    # Extract pathway source (KEGG, Hallmark, Reactome)
                    if '_' in str(pathway_name):
                        pathway_source = pathway_name.split('_')[0]
                    else:
                        pathway_source = 'Unknown'
                    
                    gsea_results_list.append({
                        'cell_type': celltype,
                        'pathway': i,
                        'pathway_name': pathway_name,
                        'pathway_source': pathway_source,
                        'nes': nes,
                        'pval': pval,
                        'fdr': fdr,
                        'leading_edge': leading_edge
                    })
                    
                    # Store in pathway lookup
                    if i not in pathway_lookup:
                        pathway_lookup[i] = {
                            'name': pathway_name,
                            'source': pathway_source,
                            'genes': gene_sets_dict.get(pathway_name, [])
                        }
                        
            except Exception as e:
                print(f"        Warning: GSEA failed for {celltype}: {e}")
                continue
                
        except Exception as e:
            print(f"        Error processing {celltype}: {e}")
            continue
    
    # Create pathway metadata lookup
    print("      Creating pathway metadata lookup...")
    gsea_df = pd.DataFrame(gsea_results_list)
    print(f"      ✓ {len(gsea_df)} total enrichments")
    
    # Save both dataframe and pathway lookup
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'gsea_df': gsea_df,
            'pathway_lookup': pathway_lookup
        }, f)
    
    # Also save pathway lookup separately for easy access
    with open(f'{CHECKPOINT_DIR}/P2_gsea_results_pathway_lookup.pkl', 'wb') as f:
        pickle.dump(pathway_lookup, f)
    print(f"      ✓ Saved pathway lookup: {CHECKPOINT_DIR}/P2_gsea_results_pathway_lookup.pkl")

print()
# VISUALIZATION 3: GSEA Results Overview
if len(gsea_df) > 0:
    print("\n  📊 Generating GSEA results visualization...")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Panel A: Enrichments per cell type
    ax1 = fig.add_subplot(gs[0, 0])
    ct_enrichments = gsea_df.groupby('cell_type').size().sort_values(ascending=False).head(20)
    colors_ct = plt.cm.viridis(np.linspace(0.2, 0.9, len(ct_enrichments)))
    ax1.barh(range(len(ct_enrichments)), ct_enrichments.values, color=colors_ct)
    ax1.set_yticks(range(len(ct_enrichments)))
    ax1.set_yticklabels([ct[:35] for ct in ct_enrichments.index], fontsize=8)
    ax1.set_xlabel('Number of Enrichments', fontweight='bold')
    ax1.set_title('A) Top 20 Cell Types by Enrichments', fontweight='bold')
    ax1.invert_yaxis()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel B: NES Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(gsea_df['nes'], bins=50, color='#667eea', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Normalized Enrichment Score (NES)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('B) NES Distribution', fontweight='bold')
    ax2.text(0.02, 0.98, f'Upregulated: {(gsea_df["nes"] > 0).sum():,}', 
            transform=ax2.transAxes, va='top', fontweight='bold', color='red')
    ax2.text(0.02, 0.92, f'Downregulated: {(gsea_df["nes"] < 0).sum():,}', 
            transform=ax2.transAxes, va='top', fontweight='bold', color='blue')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Panel C: FDR Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(np.log10(gsea_df['fdr'] + 1e-10), bins=50, color='#764ba2', alpha=0.7, edgecolor='black')
    ax3.axvline(x=np.log10(0.05), color='red', linestyle='--', linewidth=2, label='FDR = 0.05')
    ax3.axvline(x=np.log10(0.01), color='darkred', linestyle='--', linewidth=2, label='FDR = 0.01')
    ax3.set_xlabel('log10(FDR)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('C) FDR Distribution', fontweight='bold')
    ax3.legend(frameon=False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Panel D: Pathway source breakdown
    ax4 = fig.add_subplot(gs[1, 1])
    if 'pathway_source' in gsea_df.columns:
        source_counts = gsea_df['pathway_source'].value_counts()
        colors_source = ['#3498DB', '#E74C3C', '#2ECC71'][:len(source_counts)]
        wedges, texts, autotexts = ax4.pie(source_counts.values, labels=source_counts.index, 
                                            autopct='%1.1f%%', colors=colors_source, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        ax4.set_title('D) Pathway Database Composition', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Pathway source data unavailable', ha='center', va='center',
                transform=ax4.transAxes)
    
    # Panel E: Top upregulated pathways
    ax5 = fig.add_subplot(gs[2, 0])
    top_up = gsea_df.nlargest(10, 'nes')[['pathway_name', 'nes']]
    colors_up = plt.cm.Reds(np.interp(top_up['nes'].values,
                                      [top_up['nes'].min(), top_up['nes'].max()],
                                      [0.4, 0.9]))
    ax5.barh(range(len(top_up)), top_up['nes'].values, color=colors_up)
    ax5.set_yticks(range(len(top_up)))
    ax5.set_yticklabels([p[:40] for p in top_up['pathway_name']], fontsize=8)
    ax5.set_xlabel('NES', fontweight='bold')
    ax5.set_title('E) Top 10 Upregulated Pathways', fontweight='bold')
    ax5.invert_yaxis()
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # Panel F: Top downregulated pathways
    ax6 = fig.add_subplot(gs[2, 1])
    top_down = gsea_df.nsmallest(10, 'nes')[['pathway_name', 'nes']]
    colors_down = plt.cm.Blues(np.interp(np.abs(top_down['nes'].values),
                                         [np.abs(top_down['nes']).min(), np.abs(top_down['nes']).max()],
                                         [0.4, 0.9]))
    ax6.barh(range(len(top_down)), top_down['nes'].values, color=colors_down)
    ax6.set_yticks(range(len(top_down)))
    ax6.set_yticklabels([p[:40] for p in top_down['pathway_name']], fontsize=8)
    ax6.set_xlabel('NES', fontweight='bold')
    ax6.set_title('F) Top 10 Downregulated Pathways', fontweight='bold')
    ax6.invert_yaxis()
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    plt.suptitle(f'P2 GSEA Results Overview - {TISSUE.title()}', fontsize=16, fontweight='bold')
    plt.savefig(f'figures/P2_GSEA/Step6_GSEA_Overview_{TISSUE.replace(" ", "")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: Step6_GSEA_Overview.png")
# ============================================================================
# STEP 7: Filter GSEA results
# ============================================================================
print(f"[7/7] Filtering GSEA results (|NES| > {NES_THRESHOLD}, FDR < {FDR_THRESHOLD})...")
checkpoint_path = f'{CHECKPOINT_DIR}/P2_gsea_results_filtered.pkl'

if os.path.exists(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        gsea_filtered = pickle.load(f)
else:
    if len(gsea_df) > 0:
        gsea_filtered = gsea_df[
            (gsea_df['fdr'] < FDR_THRESHOLD) & 
            (abs(gsea_df['nes']) > NES_THRESHOLD)
        ].copy()
    else:
        print("      WARNING: No GSEA results to filter (all cell types failed)")
        gsea_filtered = pd.DataFrame()
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(gsea_filtered, f)

print(f"      ✓ {len(gsea_filtered)} significant enrichments")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("P2 COMPLETE - SUMMARY")
print("=" * 80)
print(f"Filtered data:     {adata.n_obs} cells, {adata.n_vars} genes")
print(f"Cell types:        {adata.obs['annotation2v2'].nunique()}")
print(f"Patients:          {adata.obs['donor_id'].nunique()}")
print(f"Total enrichments: {len(gsea_df)}")
print(f"Significant (|NES|>{NES_THRESHOLD}, FDR<{FDR_THRESHOLD}): {len(gsea_filtered)}")

if len(gsea_filtered) > 0:
    print(f"Unique pathways:   {gsea_filtered['pathway'].nunique()}")
    
    # Show pathway source breakdown
    if 'pathway_source' in gsea_filtered.columns:
        print(f"\nPathway source breakdown:")
        print(gsea_filtered['pathway_source'].value_counts())
else:
    print(f"Unique pathways:   0")
# VISUALIZATION 4: Filtered GSEA Results
if len(gsea_filtered) > 0:
    print("\n  📊 Generating filtered GSEA visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Before/After filtering
    axes[0, 0].bar(['All\\nEnrichments', 'Significant\\n(Filtered)'], 
                   [len(gsea_df), len(gsea_filtered)],
                   color=['#95A5A6', '#2ECC71'], alpha=0.8, edgecolor='black')
    axes[0, 0].set_ylabel('Number of Enrichments', fontweight='bold')
    axes[0, 0].set_title('A) Filtering Effect', fontweight='bold')
    for i, v in enumerate([len(gsea_df), len(gsea_filtered)]):
        axes[0, 0].text(i, v + max(len(gsea_df), len(gsea_filtered))*0.02,
                       f'{v:,}\\n({v/len(gsea_df)*100:.1f}%)',
                       ha='center', va='bottom', fontweight='bold')
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)
    
    # Panel B: NES vs FDR scatter
    scatter_all = axes[0, 1].scatter(gsea_df['nes'], gsea_df['fdr'], 
                                    c='#95A5A6', s=20, alpha=0.3, label='Filtered out')
    scatter_sig = axes[0, 1].scatter(gsea_filtered['nes'], gsea_filtered['fdr'],
                                    c='#2ECC71', s=30, alpha=0.7, label='Significant')
    axes[0, 1].axhline(y=FDR_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                      label=f'FDR < {FDR_THRESHOLD}')
    axes[0, 1].axvline(x=NES_THRESHOLD, color='orange', linestyle='--', linewidth=2,
                      label=f'|NES| > {NES_THRESHOLD}')
    axes[0, 1].axvline(x=-NES_THRESHOLD, color='orange', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('NES', fontweight='bold')
    axes[0, 1].set_ylabel('FDR', fontweight='bold')
    axes[0, 1].set_title('B) Filtering Thresholds', fontweight='bold')
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(alpha=0.3)
    
    # Panel C: Unique pathways per cell type
    ct_pathways = gsea_filtered.groupby('cell_type')['pathway'].nunique().sort_values(ascending=False).head(15)
    axes[1, 0].barh(range(len(ct_pathways)), ct_pathways.values,
                   color=plt.cm.plasma(np.linspace(0.2, 0.9, len(ct_pathways))))
    axes[1, 0].set_yticks(range(len(ct_pathways)))
    axes[1, 0].set_yticklabels([ct[:30] for ct in ct_pathways.index], fontsize=8)
    axes[1, 0].set_xlabel('Unique Pathways', fontweight='bold')
    axes[1, 0].set_title('C) Unique Pathways per Cell Type', fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)
    
    # Panel D: Summary table
    summary_data = [
        ['Total Enrichments', f'{len(gsea_df):,}'],
        ['Significant Enrichments', f'{len(gsea_filtered):,}'],
        ['Filtering Rate', f'{len(gsea_filtered)/len(gsea_df)*100:.1f}%'],
        ['Unique Pathways', f'{gsea_filtered["pathway"].nunique()}'],
        ['Unique Cell Types', f'{gsea_filtered["cell_type"].nunique()}'],
        ['Upregulated', f'{(gsea_filtered["nes"] > 0).sum()}'],
        ['Downregulated', f'{(gsea_filtered["nes"] < 0).sum()}']
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
    
    plt.suptitle(f'P2 Filtered GSEA Results - {TISSUE.title()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/P2_GSEA/Step7_Filtered_Results_{TISSUE.replace(" ", "")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: Step7_Filtered_Results.png")

print("=" * 80)
