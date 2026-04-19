"""
P5: Interactive Drug-Patient Recommendation System
Combines visualizations with interactive HTML patient reports
"""
import json
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_palette("husl")
# ============================================================================
# Configuration
# ============================================================================
P4_RESULTS_DIR = Path('results/P4_results')
OUTPUT_DIR = Path('figures/P5_paper_figs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'patient_reports').mkdir(exist_ok=True)

CHECKPOINT_DIR = Path('checkpoints')

# File paths
SI_RECS = P4_RESULTS_DIR / 'drug_patient_recommendations_improved.json'
COLON_RECS = P4_RESULTS_DIR / 'colon' / 'drug_patient_recommendations_improved.json'

print("="*80)
print("P5: INTERACTIVE DRUG-PATIENT RECOMMENDATION SYSTEM")
print("="*80)

# ============================================================================
# [1/8] Load Data
# ============================================================================
print("\n[1/8] Loading recommendations and data...")

# Load recommendations
with open(SI_RECS, 'r') as f:
    si_recommendations = json.load(f)

with open(COLON_RECS, 'r') as f:
    colon_recommendations = json.load(f)

# Load P3 pathway information
with open(CHECKPOINT_DIR / 'P3_pathway_embeddings.pkl', 'rb') as f:
    si_p3 = pickle.load(f)

with open(CHECKPOINT_DIR / 'colon' / 'P3_pathway_embeddings.pkl', 'rb') as f:
    colon_p3 = pickle.load(f)

# Load P2 GSEA for pathway details
si_gsea = pd.read_pickle(CHECKPOINT_DIR / 'P2_gsea_results_filtered.pkl')
colon_gsea = pd.read_pickle(CHECKPOINT_DIR / 'colon' / 'P2_gsea_results_filtered.pkl')

# Create pathway ID to name mapping
si_pathway_map = dict(zip(si_gsea['pathway'], si_gsea['pathway_name']))
colon_pathway_map = dict(zip(colon_gsea['pathway'], colon_gsea['pathway_name']))

# Load disease status from original h5ad file
adata_full = sc.read_h5ad('/Users/rvellamcheti/Downloads/Cleaned_raw_annotated_object_LK.v2.h5ad')

# Get disease status per donor for small intestine
si_data = adata_full[adata_full.obs['organ__ontology_label'] == 'small intestine'].copy()
si_patient_disease = si_data.obs.groupby('donor_id')['disease__ontology_label'].first()
si_crohns_patients = si_patient_disease[si_patient_disease == "Crohn's disease"].index.astype(str).tolist()

# Get disease status per donor for colon
colon_data = adata_full[adata_full.obs['organ__ontology_label'] == 'colon'].copy()
colon_patient_disease = colon_data.obs.groupby('donor_id')['disease__ontology_label'].first()
colon_crohns_patients = colon_patient_disease[colon_patient_disease == "Crohn's disease"].index.astype(str).tolist()

del adata_full, si_data, colon_data  # Free memory

print(f"      Small Intestine: {len(si_recommendations)} patients ({len(si_crohns_patients)} Crohn's)")
print(f"      Colon: {len(colon_recommendations)} patients ({len(colon_crohns_patients)} Crohn's)")

# ============================================================================
# [2/8] Generate Cross-Tissue Paper Visualizations
# ============================================================================
print("\n[2/8] Generating cross-tissue comparison figures...")

# Convert recommendations to dataframes for analysis
si_df_list = []
for patient_id, recs in si_recommendations.items():
    for rec in recs:
        si_df_list.append({
            'patient_id': patient_id,
            'pathway': rec['pathway'],
            'pathway_name': si_pathway_map.get(rec['pathway'], f"Pathway {rec['pathway']}"),
            'prediction_score': rec['prediction_score'],
            'pathway_activity': rec['pathway_activity'],
            'nes': rec['nes'],
            'fdr': rec['fdr']
        })

colon_df_list = []
for patient_id, recs in colon_recommendations.items():
    for rec in recs:
        colon_df_list.append({
            'patient_id': patient_id,
            'pathway': rec['pathway'],
            'pathway_name': colon_pathway_map.get(rec['pathway'], f"Pathway {rec['pathway']}"),
            'prediction_score': rec['prediction_score'],
            'pathway_activity': rec['pathway_activity'],
            'nes': rec['nes'],
            'fdr': rec['fdr']
        })

si_all_df = pd.DataFrame(si_df_list)
colon_all_df = pd.DataFrame(colon_df_list)

# Figure 1: Overall Summary Statistics
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Cross-Tissue Drug-Pathway Analysis Summary', fontsize=16, fontweight='bold')

# Panel A: Recommendation counts
ax1 = axes[0, 0]
tissues = ['Small\nIntestine', 'Colon']
n_recs = [len(si_all_df), len(colon_all_df)]
colors = ['#2E86AB', '#F18F01']

bars = ax1.bar(tissues, n_recs, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Total Pathway-Patient Pairs', fontsize=12, fontweight='bold')
ax1.set_title('A) Dataset Size', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, n_recs):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel B: Prediction score distribution
ax2 = axes[0, 1]
ax2.hist(si_all_df['prediction_score'], bins=50, alpha=0.6, label='Small Intestine',
         color='#2E86AB', edgecolor='black', density=True)
ax2.hist(colon_all_df['prediction_score'], bins=50, alpha=0.6, label='Colon',
         color='#F18F01', edgecolor='black', density=True)
ax2.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='High Confidence')
ax2.set_xlabel('Prediction Score', fontsize=12, fontweight='bold')
ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
ax2.set_title('B) Prediction Score Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Panel C: Top pathways frequency
ax3 = axes[1, 0]
si_top_pathways = si_all_df['pathway_name'].value_counts().head(10)
y_pos = np.arange(len(si_top_pathways))
ax3.barh(y_pos, si_top_pathways.values, color='#2E86AB', alpha=0.7)
ax3.set_yticks(y_pos)
ax3.set_yticklabels([name[:40] for name in si_top_pathways.index], fontsize=9)
ax3.set_xlabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('C) Top 10 Pathways (Small Intestine)', fontsize=13, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# Panel D: Pathway activity vs prediction
ax4 = axes[1, 1]
scatter1 = ax4.scatter(si_all_df['pathway_activity'], si_all_df['prediction_score'],
                      c=si_all_df['prediction_score'], cmap='Blues',
                      s=20, alpha=0.5, label='Small Intestine')
scatter2 = ax4.scatter(colon_all_df['pathway_activity'], colon_all_df['prediction_score'],
                      c=colon_all_df['prediction_score'], cmap='Oranges',
                      s=20, alpha=0.5, marker='s', label='Colon')
ax4.set_xlabel('Pathway Activity', fontsize=12, fontweight='bold')
ax4.set_ylabel('Prediction Score', fontsize=12, fontweight='bold')
ax4.set_title('D) Activity vs Prediction Score', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure1_CrossTissue_Summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ Saved: Figure1_CrossTissue_Summary.png")

# Figure 2: Patient-specific heatmaps (Board Quality)
import re

def clean_pathway_name(full_name):
    name = full_name
    for prefix in ['KEGG_', 'Reactome_', 'Hallmark_']:
        name = name.replace(prefix, '')
    name = re.sub(r'\s*R-HSA-\d+', '', name)
    name = re.sub(r'_R-HSA-\d+', '', name)
    name = name.replace('_', ' ').strip()
    return name[:45]

fig, axes = plt.subplots(1, 2, figsize=(26, 16))
fig.suptitle('Patient × Pathway Recommendation Heatmaps',
             fontsize=26, fontweight='bold', y=1.02)

# Small Intestine heatmap
si_patients = sorted(si_recommendations.keys())
si_top_pathways_list = si_all_df['pathway'].value_counts().head(15).index

si_matrix = []
si_pathway_labels = []
for pathway in si_top_pathways_list:
    row = []
    for patient in si_patients:
        patient_recs = pd.DataFrame(si_recommendations[patient])
        if pathway in patient_recs['pathway'].values:
            score = patient_recs[patient_recs['pathway'] == pathway]['prediction_score'].values[0]
            row.append(score)
        else:
            row.append(0)
    si_matrix.append(row)
    si_pathway_labels.append(clean_pathway_name(si_pathway_map.get(pathway, f"Pathway {pathway}")))

si_df_matrix = pd.DataFrame(
    si_matrix,
    index=si_pathway_labels,
    columns=[f"P{p}" for p in si_patients]
)

sns.heatmap(
    si_df_matrix,
    ax=axes[0],
    cmap='Oranges',
    vmin=0.9,
    vmax=1.0,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Prediction Score', 'shrink': 0.8},
    xticklabels=True,
    yticklabels=True
)
axes[0].set_title('Small Intestine', fontsize=20, fontweight='bold', pad=15)
axes[0].set_xlabel('Patient ID', fontsize=16, fontweight='bold', labelpad=10)
axes[0].set_ylabel('Pathway', fontsize=16, fontweight='bold', labelpad=10)
axes[0].tick_params(axis='x', labelsize=11, rotation=45)
axes[0].tick_params(axis='y', labelsize=11)

# Colon heatmap
colon_patients = sorted(colon_recommendations.keys())
colon_top_pathways_list = colon_all_df['pathway'].value_counts().head(15).index

colon_matrix = []
colon_pathway_labels = []
for pathway in colon_top_pathways_list:
    row = []
    for patient in colon_patients:
        patient_recs = pd.DataFrame(colon_recommendations[patient])
        if pathway in patient_recs['pathway'].values:
            score = patient_recs[patient_recs['pathway'] == pathway]['prediction_score'].values[0]
            row.append(score)
        else:
            row.append(0)
    colon_matrix.append(row)
    colon_pathway_labels.append(clean_pathway_name(colon_pathway_map.get(pathway, f"Pathway {pathway}")))

colon_df_matrix = pd.DataFrame(
    colon_matrix,
    index=colon_pathway_labels,
    columns=[f"P{p}" for p in colon_patients]
)

sns.heatmap(
    colon_df_matrix,
    ax=axes[1],
    cmap='Oranges',
    vmin=0.9,
    vmax=1.0,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Prediction Score', 'shrink': 0.8},
    xticklabels=True,
    yticklabels=True
)
axes[1].set_title('Colon', fontsize=20, fontweight='bold', pad=15)
axes[1].set_xlabel('Patient ID', fontsize=16, fontweight='bold', labelpad=10)
axes[1].set_ylabel('Pathway', fontsize=16, fontweight='bold', labelpad=10)
axes[1].tick_params(axis='x', labelsize=11, rotation=45)
axes[1].tick_params(axis='y', labelsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure2_Patient_Pathway_Heatmaps.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ Saved: Figure2_Patient_Pathway_Heatmaps.png")

# Figure 3: ROC Curves Side by Side (Board Quality)
from sklearn.metrics import roc_curve, auc
import pickle

# Load models and data
with open(CHECKPOINT_DIR / 'P4_improved_model.pkl', 'rb') as f:
    si_model_data = pickle.load(f)

with open(CHECKPOINT_DIR / 'colon' / 'P4_improved_model.pkl', 'rb') as f:
    colon_model_data = pickle.load(f)

si_model = si_model_data['model']
si_scaler = si_model_data['scaler']
colon_model = colon_model_data['model']
colon_scaler = colon_model_data['scaler']

# Load feature matrices
with open(CHECKPOINT_DIR / 'P4_feature_matrix.pkl', 'rb') as f:
    si_features = pickle.load(f)

with open(CHECKPOINT_DIR / 'colon' / 'P4_feature_matrix.pkl', 'rb') as f:
    colon_features = pickle.load(f)

X_si = si_scaler.transform(si_features['X'])
y_si = si_features['y']

X_colon = colon_scaler.transform(colon_features['X'])
y_colon = colon_features['y']

# Generate ROC curves
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Model Performance: ROC Curves', fontsize=26, fontweight='bold')

for ax, model, X, y, title, color_train, color_test in [
    (axes[0], si_model, X_si, y_si, 'Small Intestine', '#2E86AB', '#E84855'),
    (axes[1], colon_model, X_colon, y_colon, 'Colon', '#2E86AB', '#E84855')
]:
    y_proba = model.predict_proba(X)[:, 1]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)
    
    ax.plot(fpr_train, tpr_train, color=color_train, lw=3,
            label=f'Train (AUC = {auc_train:.3f})')
    ax.plot(fpr_test, tpr_test, color=color_test, lw=3,
            label=f'Test (AUC = {auc_test:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Chance')
    ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.legend(loc='lower right', fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure3_ROC_Curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ Saved: Figure3_ROC_Curves.png")

# Figure 4: Cross Validation Folds Side by Side (Board Quality)
from sklearn.model_selection import cross_val_score, StratifiedKFold

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Model Consistency: Cross-Validation Results', fontsize=26, fontweight='bold')

for ax, model, X, y, title, mean_auc, std_auc in [
    (axes[0], si_model, X_si, y_si, 'Small Intestine', 0.811, 0.011),
    (axes[1], colon_model, X_colon, y_colon, 'Colon', 0.767, 0.031)
]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    colors = ['#2E86AB' if s >= mean_auc else '#E84855' for s in cv_scores]
    
    bars = ax.bar(range(1, 6), cv_scores, color=colors, 
                  alpha=0.85, edgecolor='black', linewidth=1.5, width=0.6)
    
    ax.axhline(y=cv_scores.mean(), color='black', linestyle='--', 
               linewidth=2.5, label=f'Mean AUC: {cv_scores.mean():.3f}')
    ax.axhline(y=cv_scores.mean() + cv_scores.std(), color='gray', 
               linestyle=':', linewidth=2, label=f'±1 Std: {cv_scores.std():.3f}')
    ax.axhline(y=cv_scores.mean() - cv_scores.std(), color='gray', 
               linestyle=':', linewidth=2)
    
    for bar, score in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Fold Number', fontsize=16, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.legend(loc='lower right', fontsize=13)
    ax.set_ylim([0.5, 1.0])
    ax.set_xticks(range(1, 6))
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure4_CrossValidation.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ Saved: Figure4_CrossValidation.png")

# Figure 5: Pathway Similarity UMAP (Board Quality)
print("\n  Creating Figure 5: Pathway Similarity UMAP...")
import umap as umap_lib

si_embeddings = si_p3['healthy_embeddings']
colon_embeddings = colon_p3['healthy_embeddings']

si_norms = np.linalg.norm(si_embeddings, axis=1)
colon_norms = np.linalg.norm(colon_embeddings, axis=1)

print("      Running UMAP for Small Intestine...")
si_reducer = umap_lib.UMAP(n_components=2, random_state=42)
si_coords = si_reducer.fit_transform(si_embeddings)

print("      Running UMAP for Colon...")
colon_reducer = umap_lib.UMAP(n_components=2, random_state=42)
colon_coords = colon_reducer.fit_transform(colon_embeddings)

fig, axes = plt.subplots(1, 2, figsize=(22, 12))
fig.suptitle('Pathway Similarity Clusters (SVD Embeddings)', fontsize=22, fontweight='bold')

for ax, coords, norms, title in [
    (axes[0], si_coords, si_norms, 'Small Intestine'),
    (axes[1], colon_coords, colon_norms, 'Colon')
]:
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=norms,
        cmap='viridis',
        s=80,
        alpha=0.75,
        edgecolors='black',
        linewidths=0.4
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Pathway Embedding Strength', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=11)
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    x_pad = x_range * 0.1
    y_pad = y_range * 0.1
    ax.set_xlim(coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
    ax.set_ylim(coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)

plt.subplots_adjust(wspace=0.35)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Figure5_UMAP.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ Saved: Figure5_UMAP.png")

# ============================================================================
# [3/8] Generate HTML Patient Report Function
# ============================================================================
print("\n[3/8] Setting up patient report generator...")

def generate_patient_report(patient_id, tissue_name, recommendations_dict, pathway_map, tissue_color, all_patients_list):
    """Generate detailed HTML report with dropdown menu for patient selection"""
    
    patient_recs = recommendations_dict[patient_id]
    rec_df = pd.DataFrame(patient_recs)
    rec_df['pathway_name'] = rec_df['pathway'].apply(lambda x: pathway_map.get(x, f"Pathway {x}"))
    rec_df = rec_df.sort_values('prediction_score', ascending=False)
    
    # Statistics
    n_total = len(rec_df)
    n_high_conf = (rec_df['prediction_score'] > 0.7).sum()
    n_med_conf = ((rec_df['prediction_score'] > 0.5) & (rec_df['prediction_score'] <= 0.7)).sum()
    avg_score = rec_df['prediction_score'].mean()
    
    # Generate dropdown options
    dropdown_options = ""
    for pid in all_patients_list:
        selected = "selected" if pid == patient_id else ""
        dropdown_options += f'                    <option value="{pid}" {selected}>Patient {pid}</option>\n'
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Patient Report - {patient_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, {tissue_color} 0%, #667eea 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 5px 0 0 0;
            font-size: 16px;
            opacity: 0.9;
        }}
        .dropdown-section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .dropdown-section h2 {{
            color: {tissue_color};
            margin: 0 0 20px 0;
        }}
        select {{
            padding: 12px 20px;
            font-size: 16px;
            border: 2px solid {tissue_color};
            border-radius: 5px;
            background: white;
            cursor: pointer;
            min-width: 300px;
            font-weight: 600;
        }}
        select:hover {{
            background: #f9f9f9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: {tissue_color};
            border-bottom: 3px solid {tissue_color};
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, {tissue_color} 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-box .number {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-box .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .drug-card {{
            background: #f9f9f9;
            border-left: 4px solid {tissue_color};
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .drug-card h3 {{
            margin: 0 0 10px 0;
            color: {tissue_color};
        }}
        .drug-details {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 10px;
            font-size: 14px;
        }}
        .drug-detail {{
            background: white;
            padding: 8px;
            border-radius: 4px;
        }}
        .drug-detail strong {{
            color: {tissue_color};
        }}
        .score-bar {{
            display: inline-block;
            height: 20px;
            background: linear-gradient(90deg, #667eea 0%, {tissue_color} 100%);
            border-radius: 10px;
            margin-right: 10px;
        }}
        .high-conf {{ color: #06A77D; font-weight: bold; }}
        .med-conf {{ color: #F18F01; font-weight: bold; }}
        .low-conf {{ color: #999; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background-color: {tissue_color};
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 14px;
        }}
    </style>
    <script>
        function changePatient() {{
            var select = document.getElementById("patientSelect");
            var patientId = select.value;
            var tissue = "{tissue_name}".replace(" ", "");
            window.location.href = "Patient_" + patientId + "_" + tissue + ".html";
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>🧬 Personalized Drug Recommendation Report</h1>
        <p>Tissue: <strong>{tissue_name}</strong> | Generated: February 2026</p>
    </div>

    <div class="dropdown-section">
        <h2>🔍 Select Patient to View Report</h2>
        <select id="patientSelect" onchange="changePatient()">
{dropdown_options}
        </select>
        <p style="margin-top: 15px; color: #666; font-size: 14px;">
            Currently viewing: <strong>Patient {patient_id}</strong>
        </p>
    </div>

    <div class="section">
        <h2>📊 Patient Profile Summary</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="label">Total Pathway Recommendations</div>
                <div class="number">{n_total}</div>
            </div>
            <div class="stat-box">
                <div class="label">High Confidence (&gt;0.7)</div>
                <div class="number">{n_high_conf}</div>
            </div>
            <div class="stat-box">
                <div class="label">Medium Confidence (0.5-0.7)</div>
                <div class="number">{n_med_conf}</div>
            </div>
            <div class="stat-box">
                <div class="label">Average Match Score</div>
                <div class="number">{avg_score:.3f}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>💊 Top 5 Personalized Drug-Pathway Recommendations</h2>
        <p>These are the highest-confidence therapeutic pathways identified for this patient.</p>
"""
    
    # Add top 5 drug cards
    for idx, row in rec_df.head(5).iterrows():
        pathway_name = row['pathway_name']
        if len(pathway_name) > 100:
            pathway_name = pathway_name[:97] + "..."
        
        score = row['prediction_score']
        nes = row['nes']
        fdr = row['fdr']
        pathway_activity = row['pathway_activity']
        
        if score > 0.7:
            conf_class, conf_label = "high-conf", "⭐ HIGH CONFIDENCE"
        elif score > 0.5:
            conf_class, conf_label = "med-conf", "⚡ MEDIUM CONFIDENCE"
        else:
            conf_class, conf_label = "low-conf", "◐ LOW CONFIDENCE"
        
        bar_width = int(score * 300)
        
        if pathway_activity > 0.3 and nes > 0:
            interpretation = "High activity + upregulation suggests a key pathology driver."
        elif pathway_activity < 0.1 and nes < 0:
            interpretation = "Low activity + downregulation indicates potential therapeutic restoration target."
        elif pathway_activity > 0.3 and nes < 0:
            interpretation = "High activity despite population downregulation suggests patient-specific dysregulation."
        else:
            interpretation = "Moderate dysregulation pattern that may benefit from therapeutic intervention."
        
        html += f"""
        <div class="drug-card">
            <h3>#{idx+1}. {pathway_name}</h3>
            <p><span class="{conf_class}">{conf_label}</span> | Match Score: <strong>{score:.3f}</strong></p>
            <div class="score-bar" style="width: {bar_width}px;"></div>
            <span>{score*100:.1f}%</span>
            
            <div class="drug-details">
                <div class="drug-detail">
                    <strong>Pathway Activity:</strong><br>{pathway_activity:.4f}
                </div>
                <div class="drug-detail">
                    <strong>Population NES:</strong><br>{nes:.2f} ({'Up' if nes > 0 else 'Down'}-regulated)
                </div>
                <div class="drug-detail">
                    <strong>Statistical FDR:</strong><br>{fdr:.4f}
                </div>
                <div class="drug-detail">
                    <strong>Confidence:</strong><br>{conf_label.replace('⭐ ', '').replace('⚡ ', '').replace('◐ ', '')}
                </div>
            </div>
            
            <p style="margin-top: 15px; padding: 10px; background: white; border-radius: 5px; font-size: 14px;">
                <strong>🔍 Interpretation:</strong> {interpretation}
            </p>
        </div>
"""
    
    html += f"""
    </div>

    <div class="section">
        <h2>📋 Complete Top 20 Recommendations</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Pathway/Drug Target</th>
                    <th>Match Score</th>
                    <th>Activity</th>
                    <th>NES</th>
                    <th>FDR</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for idx, row in enumerate(rec_df.head(20).itertuples(), 1):
        pathway_name = row.pathway_name
        if len(pathway_name) > 60:
            pathway_name = pathway_name[:57] + "..."
        
        score = row.prediction_score
        if score > 0.7:
            score_class = "high-conf"
        elif score > 0.5:
            score_class = "med-conf"
        else:
            score_class = "low-conf"
        
        html += f"""
                <tr>
                    <td><strong>{idx}</strong></td>
                    <td>{pathway_name}</td>
                    <td class="{score_class}">{score:.3f}</td>
                    <td>{row.pathway_activity:.4f}</td>
                    <td>{row.nes:.2f}</td>
                    <td>{row.fdr:.4f}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>ℹ️ How to Interpret This Report</h2>
        <p><strong>Match Score:</strong> Probability (0-1) that this pathway is a good therapeutic target for this patient.</p>
        <p><strong>NES:</strong> Normalized Enrichment Score - pathway dysregulation in Crohn's population (+ = up, - = down).</p>
        <p><strong>FDR:</strong> False Discovery Rate - statistical confidence (&lt;0.25 = significant).</p>
        <p><strong>Pathway Activity:</strong> How active this pathway is in this patient's cells specifically.</p>
    </div>

    <div class="footer">
        <p><strong>Disclaimer:</strong> For research purposes only. Not for clinical use without medical validation.</p>
        <p>AI-Powered Drug-Patient Matching System | Single-cell RNA-seq Analysis</p>
    </div>
</body>
</html>
"""
    
    return html

print("      ✓ Patient report generator ready")

# ============================================================================
# [4/8] Generate Small Intestine Patient Reports
# ============================================================================
print("\n[4/8] Generating Small Intestine patient reports (Crohn's only)...")

# Filter for Crohn's patients
si_crohns_recs = {k: v for k, v in si_recommendations.items() if k in si_crohns_patients}

# Get top 5 patients by high-confidence matches
si_patient_scores = {}
for patient_id, recs in si_crohns_recs.items():
    df = pd.DataFrame(recs)
    n_high_conf = (df['prediction_score'] > 0.7).sum()
    si_patient_scores[patient_id] = n_high_conf

si_top_patients = sorted(si_patient_scores.items(), key=lambda x: x[1], reverse=True)[:5]
si_top_patient_ids = [p[0] for p in si_top_patients]

si_count = 0
for patient_id in si_crohns_patients:
    html = generate_patient_report(
        patient_id, 
        "Small Intestine", 
        si_recommendations,
        si_pathway_map,
        "#285669",
        si_crohns_patients
    )
    
    filename = OUTPUT_DIR / 'patient_reports' / f'Patient_{patient_id}_SmallIntestine.html'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    si_count += 1

print(f"      ✓ Generated {si_count} Small Intestine reports")
print(f"      Top patients: {', '.join(si_top_patient_ids)}")

# ============================================================================
# [5/8] Generate Colon Patient Reports
# ============================================================================
print("\n[5/8] Generating Colon patient reports (Crohn's only)...")

# Filter for Crohn's patients
colon_crohns_recs = {k: v for k, v in colon_recommendations.items() if k in colon_crohns_patients}

# Get top 5 patients
colon_patient_scores = {}
for patient_id, recs in colon_crohns_recs.items():
    df = pd.DataFrame(recs)
    n_high_conf = (df['prediction_score'] > 0.7).sum()
    colon_patient_scores[patient_id] = n_high_conf

colon_top_patients = sorted(colon_patient_scores.items(), key=lambda x: x[1], reverse=True)[:5]
colon_top_patient_ids = [p[0] for p in colon_top_patients]

colon_count = 0
for patient_id in colon_crohns_patients:
    html = generate_patient_report(
        patient_id,
        "Colon",
        colon_recommendations,
        colon_pathway_map,
        "#F18F01",
        colon_crohns_patients
    )
    
    filename = OUTPUT_DIR / 'patient_reports' / f'Patient_{patient_id}_Colon.html'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    colon_count += 1

print(f"      ✓ Generated {colon_count} Colon reports")
print(f"      Top patients: {', '.join(colon_top_patient_ids)}")

# ============================================================================
# [6/8] Generate Summary Statistics Table
# ============================================================================
print("\n[6/8] Generating summary statistics table...")

summary_data = {
    'Metric': [
        'Total Patients (Crohn\'s)',
        'Total Pathway-Patient Pairs',
        'Unique Pathways',
        'Average Predictions per Patient',
        'High Confidence Matches (>0.7)',
        'Medium Confidence (0.5-0.7)',
        'Mean Prediction Score',
        'Top Pathway (Most Frequent)',
        'Top Pathway Frequency'
    ],
    'Small Intestine': [
        len(si_crohns_patients),
        len(si_all_df),
        si_all_df['pathway'].nunique(),
        f"{len(si_all_df) / len(si_crohns_patients):.1f}",
        f"{(si_all_df['prediction_score'] > 0.7).sum()} ({(si_all_df['prediction_score'] > 0.7).sum()/len(si_all_df)*100:.1f}%)",
        f"{((si_all_df['prediction_score'] > 0.5) & (si_all_df['prediction_score'] <= 0.7)).sum()}",
        f"{si_all_df['prediction_score'].mean():.3f}",
        si_all_df['pathway_name'].value_counts().index[0][:40],
        si_all_df['pathway_name'].value_counts().values[0]
    ],
    'Colon': [
        len(colon_crohns_patients),
        len(colon_all_df),
        colon_all_df['pathway'].nunique(),
        f"{len(colon_all_df) / len(colon_crohns_patients):.1f}",
        f"{(colon_all_df['prediction_score'] > 0.7).sum()} ({(colon_all_df['prediction_score'] > 0.7).sum()/len(colon_all_df)*100:.1f}%)",
        f"{((colon_all_df['prediction_score'] > 0.5) & (colon_all_df['prediction_score'] <= 0.7)).sum()}",
        f"{colon_all_df['prediction_score'].mean():.3f}",
        colon_all_df['pathway_name'].value_counts().index[0][:40],
        colon_all_df['pathway_name'].value_counts().values[0]
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(OUTPUT_DIR / 'Summary_Statistics_Table.csv', index=False)

print("\n" + summary_df.to_string(index=False))
print(f"\n      ✓ Saved: Summary_Statistics_Table.csv")

# ============================================================================
# [7/8] Generate Index HTML (Dropdown Landing Page)
# ============================================================================
print("\n[7/8] Generating index page with dropdown menu...")

index_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Drug-Patient Recommendation System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: white;
            padding: 50px;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            max-width: 600px;
            text-align: center;
        }
        h1 {
            color: #667eea;
            margin: 0 0 20px 0;
            font-size: 36px;
        }
        p {
            color: #666;
            margin-bottom: 30px;
            font-size: 16px;
        }
        .tissue-section {
            background: #f9f9f9;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .tissue-section h2 {
            margin: 0 0 15px 0;
            font-size: 24px;
        }
        select {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #667eea;
            border-radius: 8px;
            cursor: pointer;
            margin-bottom: 15px;
            font-weight: 600;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: scale(1.05);
        }
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        .stat-box {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .stat-box .number {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-box .label {
            font-size: 12px;
            color: #666;
        }
    </style>
    <script>
        function viewSIReport() {
            var select = document.getElementById("siSelect");
            var patientId = select.value;
            window.location.href = "patient_reports/Patient_" + patientId + "_SmallIntestine.html";
        }
        
        function viewColonReport() {
            var select = document.getElementById("colonSelect");
            var patientId = select.value;
            window.location.href = "patient_reports/Patient_" + patientId + "_Colon.html";
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>🧬 Drug-Patient Recommendation System</h1>
        <p>AI-powered personalized therapeutic pathway recommendations for Crohn's disease patients</p>
        
        <div class="tissue-section" style="border-top: 4px solid #285669;">
            <h2 style="color: #285669;">📍 Small Intestine</h2>
            <select id="siSelect">
"""

for patient_id in si_crohns_patients:
    n_high = si_patient_scores.get(patient_id, 0)
    # Also get the max score for this patient
    patient_df = pd.DataFrame(si_recommendations[patient_id])
    max_score = patient_df['prediction_score'].max()
    index_html += f'                <option value="{patient_id}">Patient {patient_id} (top score: {max_score:.3f})</option>\n'

index_html += f"""            </select>
            <button onclick="viewSIReport()" style="background: linear-gradient(135deg, #285669 0%, #667eea 100%);">View Report</button>
            <div class="stats">
                <div class="stat-box">
                    <div class="number">{len(si_crohns_patients)}</div>
                    <div class="label">Patients</div>
                </div>
                <div class="stat-box">
                    <div class="number">{len(si_all_df):,}</div>
                    <div class="label">Predictions</div>
                </div>
            </div>
        </div>
        
        <div class="tissue-section" style="border-top: 4px solid #F18F01;">
            <h2 style="color: #F18F01;">📍 Colon</h2>
            <select id="colonSelect">
"""

for patient_id in colon_crohns_patients:
    n_high = colon_patient_scores.get(patient_id, 0)
    # Also get the max score for this patient
    patient_df = pd.DataFrame(colon_recommendations[patient_id])
    max_score = patient_df['prediction_score'].max()
    index_html += f'                <option value="{patient_id}">Patient {patient_id} (top score: {max_score:.3f})</option>\n'

index_html += f"""            </select>
            <button onclick="viewColonReport()" style="background: linear-gradient(135deg, #F18F01 0%, #764ba2 100%);">View Report</button>
            <div class="stats">
                <div class="stat-box" style="border-left-color: #F18F01;">
                    <div class="number" style="color: #F18F01;">{len(colon_crohns_patients)}</div>
                    <div class="label">Patients</div>
                </div>
                <div class="stat-box" style="border-left-color: #F18F01;">
                    <div class="number" style="color: #F18F01;">{len(colon_all_df):,}</div>
                    <div class="label">Predictions</div>
                </div>
            </div>
        </div>
        
        <p style="margin-top: 30px; font-size: 14px; color: #999;">
            Generated: February 2026 | For research purposes only
        </p>
    </div>
</body>
</html>
"""

with open(OUTPUT_DIR / 'index.html', 'w', encoding='utf-8') as f:
    f.write(index_html)

print("      ✓ Saved: index.html (landing page with dropdown)")

# ============================================================================
# [8/8] Final Summary
# ============================================================================
print("\n" + "="*80)
print("P5 COMPLETE - ALL VISUALIZATIONS AND REPORTS GENERATED!")
print("="*80)

print("\n✓ Paper Figures:")
print("  • Figure1_CrossTissue_Summary.png")
print("  • Figure2_Patient_Pathway_Heatmaps.png")

print(f"\n✓ Patient Reports (with dropdown navigation):")
print(f"  • Small Intestine: {si_count} reports")
print(f"  • Colon: {colon_count} reports")
print(f"  • Total: {si_count + colon_count} HTML reports")

print("\n✓ Summary Tables:")
print("  • Summary_Statistics_Table.csv")

print(f"\n✓ All files saved to: {OUTPUT_DIR}/")

print("\n" + "="*80)
print("HOW TO USE:")
print("="*80)
print("1. Open: figures/P5_paper_figs/index.html")
print("2. Select tissue type (Small Intestine or Colon)")
print("3. Choose patient from dropdown menu")
print("4. Click 'View Report' to see personalized recommendations")
print("5. Use dropdown in individual reports to switch between patients")
print("="*80)

print("\nKEY STATISTICS:")
print(f"  • Small Intestine: {len(si_crohns_patients)} Crohn's patients, {(si_all_df['prediction_score'] > 0.7).sum()} high-confidence matches")
print(f"  • Colon: {len(colon_crohns_patients)} Crohn's patients, {(colon_all_df['prediction_score'] > 0.7).sum()} high-confidence matches")
print("="*80)