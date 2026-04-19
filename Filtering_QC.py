"""
P1: Data Exploration for CD Fibrosis Study
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_palette("husl")

print("="*80)
print("P1: DATA EXPLORATION WITH VISUALIZATIONS")
print("="*80)

# Configure plotting
sc.settings._set_figure_params(dpi=300, facecolor="white")
plt.rcParams['figure.dpi'] = 300
sns.set_palette("husl")

# Create output directories
Path("figures/P1_figs").mkdir(parents=True, exist_ok=True)

# Load data
print("\n[1/5] Loading data...")
adata = sc.read_h5ad("/Users/rvellamcheti/Downloads/Cleaned_raw_annotated_object_LK.v2.h5ad")
print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

# Preserve raw counts and log-transform (fixes warning)
if adata.raw is None:
    adata.raw = adata.copy()
    sc.pp.log1p(adata)
    print("Log-transformed data (raw counts preserved)")

# Check CD patients
print("\n[2/5] Analyzing CD patients...")
adata_crohns = adata[adata.obs['disease'].isin(["MONDO_0005011"]), :].copy()

print(f"Total CD patients: {adata_crohns.obs['donor_id'].nunique()}")
print("\nCells per donor:")
print(adata_crohns.obs['donor_id'].value_counts().head(10))

# Generate scanpy UMAPs
print("\n[3/5] Generating scanpy UMAPs...")
sc.pl.umap(adata, color="disease", save="_P1_disease.png", show=False)
sc.pl.umap(adata, color="tissue", save="_P1_tissue.png", show=False)
sc.pl.umap(adata, color="organ", save="_P1_organ.png", show=False)
sc.pl.umap(adata, color="annotation2v2", save="_P1_celltypes.png", show=False)
sc.pl.umap(adata, color="species", save="_P1_species.png", show=False)
print("✓ Scanpy UMAPs saved to figures/")

# ============================================================================
# COMPREHENSIVE QC VISUALIZATIONS
# ============================================================================
print("\n[4/5] Generating comprehensive QC visualizations...")

# Map disease codes to readable labels
disease_map = {
    'MONDO_0005011': "Crohn's",
    'PATO_0000461': 'Healthy'
}
adata.obs['disease_label'] = (
    adata.obs['disease']
    .map(disease_map)
    .astype(str)
    .fillna('Other')
)

# Figure 1: Dataset Overview
print("  Creating Figure 1: Dataset Overview...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Panel A: Cell distribution by disease
ax1 = fig.add_subplot(gs[0, 0])
disease_counts = adata.obs['disease_label'].value_counts()
colors_disease = ['#E74C3C' if 'Crohn' in x else '#3498DB' if 'Healthy' in x else '#95A5A6' 
                  for x in disease_counts.index]
bars = ax1.bar(range(len(disease_counts)), disease_counts.values, color=colors_disease)
ax1.set_xticks(range(len(disease_counts)))
ax1.set_xticklabels(disease_counts.index, rotation=0)
ax1.set_ylabel('Number of Cells', fontweight='bold')
ax1.set_title('A) Cell Distribution by Disease', fontweight='bold', pad=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for i, v in enumerate(disease_counts.values):
    ax1.text(i, v + max(disease_counts.values)*0.02, f'{v:,}', 
            ha='center', va='bottom', fontweight='bold')

# Panel B: Patient distribution by disease
ax2 = fig.add_subplot(gs[0, 1])
patient_disease = adata.obs.groupby('donor_id')['disease_label'].first()
patient_counts = patient_disease.value_counts()
bars = ax2.bar(range(len(patient_counts)), patient_counts.values, color=colors_disease[:len(patient_counts)])
ax2.set_xticks(range(len(patient_counts)))
ax2.set_xticklabels(patient_counts.index, rotation=0)
ax2.set_ylabel('Number of Patients', fontweight='bold')
ax2.set_title('B) Patient Distribution by Disease', fontweight='bold', pad=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for i, v in enumerate(patient_counts.values):
    ax2.text(i, v + max(patient_counts.values)*0.05, f'{v}', 
            ha='center', va='bottom', fontweight='bold')

# Panel C: Top 15 cell types
ax3 = fig.add_subplot(gs[0, 2])
celltype_counts = adata.obs['annotation2v2'].value_counts().head(15)
y_pos = np.arange(len(celltype_counts))
colors_ct = plt.cm.viridis(np.linspace(0.2, 0.9, len(celltype_counts)))
ax3.barh(y_pos, celltype_counts.values, color=colors_ct)
ax3.set_yticks(y_pos)
ax3.set_yticklabels([label[:30] for label in celltype_counts.index], fontsize=8)
ax3.set_xlabel('Number of Cells', fontweight='bold')
ax3.set_title('C) Top 15 Cell Types', fontweight='bold', pad=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.invert_yaxis()

# Panel D: UMI counts distribution
ax4 = fig.add_subplot(gs[1, 0])
for condition, color in zip(["Crohn's", 'Healthy'], ['#E74C3C', '#3498DB']):
    mask = adata.obs['disease_label'] == condition
    if mask.sum() > 0:
        ax4.hist(np.log10(adata.obs.loc[mask, 'total_counts']), 
                bins=50, alpha=0.6, label=condition, color=color)
ax4.set_xlabel('log10(Total UMI Counts)', fontweight='bold')
ax4.set_ylabel('Number of Cells', fontweight='bold')
ax4.set_title('D) UMI Count Distribution', fontweight='bold', pad=10)
ax4.legend(frameon=False)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Panel E: Mitochondrial content
ax5 = fig.add_subplot(gs[1, 1])
for condition, color in zip(["Crohn's", 'Healthy'], ['#E74C3C', '#3498DB']):
    mask = adata.obs['disease_label'] == condition
    if mask.sum() > 0:
        ax5.hist(adata.obs.loc[mask, 'pct_counts_mt'], 
                bins=50, alpha=0.6, label=condition, color=color)
ax5.set_xlabel('% Mitochondrial Counts', fontweight='bold')
ax5.set_ylabel('Number of Cells', fontweight='bold')
ax5.set_title('E) Mitochondrial Content', fontweight='bold', pad=10)
ax5.legend(frameon=False)
ax5.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='QC threshold')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# Panel F: Doublet scores
ax6 = fig.add_subplot(gs[1, 2])
for condition, color in zip(["Crohn's", 'Healthy'], ['#E74C3C', '#3498DB']):
    mask = adata.obs['disease_label'] == condition
    if mask.sum() > 0:
        ax6.hist(adata.obs.loc[mask, 'doublet_score'], 
                bins=50, alpha=0.6, label=condition, color=color)
ax6.set_xlabel('Doublet Score', fontweight='bold')
ax6.set_ylabel('Number of Cells', fontweight='bold')
ax6.set_title('F) Doublet Detection', fontweight='bold', pad=10)
ax6.legend(frameon=False)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

# Panel G: Cells per patient
ax7 = fig.add_subplot(gs[2, :])
patient_cells = adata.obs.groupby('donor_id').size().sort_values(ascending=False).head(30)
patient_disease_map = adata.obs.groupby('donor_id')['disease_label'].first()
colors_patients = ['#E74C3C' if patient_disease_map[p] == "Crohn's" else '#3498DB' 
                   if patient_disease_map[p] == 'Healthy' else '#95A5A6'
                   for p in patient_cells.index]

bars = ax7.bar(range(len(patient_cells)), patient_cells.values, color=colors_patients)
ax7.set_xlabel('Patient ID (Top 30)', fontweight='bold')
ax7.set_ylabel('Number of Cells', fontweight='bold')
ax7.set_title('G) Cell Count Distribution Across Patients', fontweight='bold', pad=10)
ax7.set_xticks(range(len(patient_cells)))
ax7.set_xticklabels(patient_cells.index, rotation=90, fontsize=7)
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', label="Crohn's"),
    Patch(facecolor='#3498DB', label='Healthy')
]
ax7.legend(handles=legend_elements, loc='upper right', frameon=False)

plt.suptitle('P1 Data Overview & Quality Control', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('figures/P1_figs/Figure1_DataOverview_QC.png', bbox_inches='tight', dpi=300)
plt.close()
print("  ✓ Saved: Figure1_DataOverview_QC.png")

# Figure 2: Tissue & Organ Distribution
print("  Creating Figure 2: Tissue/Organ Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Tissue distribution
tissue_counts = adata.obs['tissue'].value_counts().head(10)
axes[0].barh(range(len(tissue_counts)), tissue_counts.values, 
            color=plt.cm.Set3(np.linspace(0, 1, len(tissue_counts))))
axes[0].set_yticks(range(len(tissue_counts)))
axes[0].set_yticklabels(tissue_counts.index, fontsize=9)
axes[0].set_xlabel('Number of Cells', fontweight='bold')
axes[0].set_title('A) Top 10 Tissues', fontweight='bold')
axes[0].invert_yaxis()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Organ distribution  
organ_counts = adata.obs['organ'].value_counts().head(10)
axes[1].barh(range(len(organ_counts)), organ_counts.values,
            color=plt.cm.Set2(np.linspace(0, 1, len(organ_counts))))
axes[1].set_yticks(range(len(organ_counts)))
axes[1].set_yticklabels(organ_counts.index, fontsize=9)
axes[1].set_xlabel('Number of Cells', fontweight='bold')
axes[1].set_title('B) Top 10 Organs', fontweight='bold')
axes[1].invert_yaxis()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('P1 Tissue & Organ Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/P1_figs/Figure2_Tissue_Organ_Distribution.png', bbox_inches='tight', dpi=300)
plt.close()
print("  ✓ Saved: Figure2_Tissue_Organ_Distribution.png")

# ============================================================================
# EXPORT SUMMARY STATISTICS
# ============================================================================
print("\n[5/5] Exporting summary statistics...")

# Overall summary
summary_stats = {
    'Total Cells': adata.n_obs,
    'Total Genes': adata.n_vars,
    'Total Patients': adata.obs['donor_id'].nunique(),
    "Crohn's Patients": (patient_disease == "Crohn's").sum(),
    'Healthy Patients': (patient_disease == 'Healthy').sum(),
    'Total Cell Types': adata.obs['annotation2v2'].nunique(),
    'Tissues': adata.obs['tissue'].nunique(),
    'Organs': adata.obs['organ'].nunique()
}

summary_df = pd.DataFrame([summary_stats]).T
summary_df.columns = ['Value']
summary_df.to_csv('figures/P1_figs/Summary_Statistics.csv')
print("  ✓ Saved: Summary_Statistics.csv")

print("\n" + "="*80)
print("P1 COMPLETE - SUMMARY")
print("="*80)
print(f"Total cells: {adata.n_obs:,}")
print(f"Total genes: {adata.n_vars:,}")
print(f"Total patients: {adata.obs['donor_id'].nunique()}")
print(f"Crohn's patients: {(patient_disease == 'Crohn\'s').sum()}")




print(f"Healthy patients: {(patient_disease == 'Healthy').sum()}")
print(f"\nFigures saved to: figures/P1_figs/")
print("="*80)

# Save processed data
adata.write("/Users/rvellamcheti/Downloads/Cleaned_raw_annotated_object_LK.v2.h5ad")
print("Processed data saved")