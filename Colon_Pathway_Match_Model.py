"""
P4: Drug-Patient Matching with Random Forest - COLON
Inputs: P2 GSEA results, P3 pathway embeddings
Outputs: Per-patient drug-pathway recommendations ranked by prediction score
"""

import numpy as np
import pandas as pd
import pickle
import scanpy as sc
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import json

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_palette("husl")

# Configuration
CHECKPOINT_DIR = Path('checkpoints/colon')
RESULTS_DIR = Path('results/P4_results/colon')
FIGURES_DIR = Path('figures/P4_Predictions')
ADATA_FILE = '/Users/rvellamcheti/Downloads/Cleaned_raw_annotated_object_LK.v2.h5ad'
P2_FILE = CHECKPOINT_DIR / 'P2_gsea_results_filtered.pkl'
P3_FILE = CHECKPOINT_DIR / 'P3_pathway_embeddings.pkl'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Random Forest hyperparameters
N_ESTIMATORS = 500
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = 'sqrt'
RANDOM_STATE = 42

print("="*80)
print("P4: DRUG-PATIENT MATCHING - COLON")
print("="*80)

# ============================================================================
# STEP 1: Load patient metadata from colon
# ============================================================================
print("\n[1/7] Loading patient information...")
adata_full = sc.read_h5ad(ADATA_FILE)
adata = adata_full[adata_full.obs['organ__ontology_label'] == 'colon'].copy()

agg_dict = {
    'disease__ontology_label': 'first',
    'sex': 'first',
    'tissue': 'first',
    'annotation': lambda x: x.value_counts().to_dict()
}
patient_metadata = adata.obs.groupby('donor_id').agg(agg_dict).reset_index()
patient_metadata['is_crohns'] = patient_metadata['disease__ontology_label'] == "Crohn's disease"

healthy_patients = patient_metadata[~patient_metadata['is_crohns']]['donor_id'].tolist()
crohns_patients = patient_metadata[patient_metadata['is_crohns']]['donor_id'].tolist()

print(f"      Healthy: {len(healthy_patients)}, Crohn's: {len(crohns_patients)}")

# ============================================================================
# STEP 2: Load P2 GSEA results and P3 pathway embeddings
# ============================================================================
print("\n[2/7] Loading GSEA results and pathway embeddings...")
gsea_results = pd.read_pickle(P2_FILE)

with open(P3_FILE, 'rb') as f:
    p3_data = pickle.load(f)

pathway_embeddings = p3_data['healthy_embeddings']
pathway_lookup = p3_data['pathway_lookup']
pathway_order = p3_data['pathway_order']
pathway_to_idx = {pathway: idx for idx, pathway in enumerate(pathway_order)}

print(f"      Pathways: {len(pathway_lookup)}")

# ============================================================================
# STEP 3: Build feature matrix
# Each row = one patient x pathway combination
# Features combine pathway embedding with GSEA-derived activity scores
# ============================================================================
print("\n[3/7] Building feature matrix...")
unique_pathways = sorted([p for p in gsea_results['pathway'].unique() if p in pathway_to_idx])

features_list = []
labels_list = []
metadata_list = []

for _, patient_row in patient_metadata.iterrows():
    patient_id = patient_row['donor_id']
    is_crohns = patient_row['is_crohns']

    # Cell type proportions used to weight pathway activity by cell abundance
    cell_type_counts = adata[adata.obs['donor_id'] == patient_id].obs['annotation'].value_counts()
    total_cells = cell_type_counts.sum()
    cell_type_proportions = (cell_type_counts / total_cells).to_dict()

    for pathway_id in unique_pathways:
        if pathway_id not in pathway_to_idx:
            continue

        pathway_idx = pathway_to_idx[pathway_id]
        pathway_emb = pathway_embeddings[pathway_idx]

        pathway_gsea = gsea_results[gsea_results['pathway'] == pathway_id]
        pathway_name = pathway_gsea['pathway_name'].iloc[0] if len(pathway_gsea) > 0 else f"pathway_{pathway_id}"

        # Weighted pathway activity: NES weighted by cell type proportion
        pathway_activity = 0
        for _, gsea_row in pathway_gsea.iterrows():
            cell_type = gsea_row['cell_type']
            nes = gsea_row['nes']
            proportion = cell_type_proportions.get(cell_type, 0)
            pathway_activity += abs(nes) * proportion

        mean_nes = pathway_gsea['nes'].mean()
        mean_fdr = pathway_gsea['fdr'].mean()
        population_signature = mean_nes
        cosine_similarity = np.linalg.norm(pathway_emb)
        relative_activity = abs(mean_nes)

        features = np.concatenate([
            pathway_emb,
            np.array([pathway_activity, population_signature, cosine_similarity,
                     mean_nes, mean_fdr, relative_activity])
        ])

        features_list.append(features)
        labels_list.append(1 if is_crohns else 0)
        metadata_list.append({
            'patient_id': patient_id,
            'pathway': pathway_name,
            'pathway_id': pathway_id,
            'is_crohns': is_crohns,
            'pathway_activity': pathway_activity,
            'nes': mean_nes,
            'fdr': mean_fdr
        })

X = np.array(features_list)
y = np.array(labels_list)
metadata = pd.DataFrame(metadata_list)

# Save feature matrix for P5 (used for ROC and cross-validation figures)
with open(CHECKPOINT_DIR / 'P4_feature_matrix.pkl', 'wb') as f:
    pickle.dump({'X': X, 'y': y}, f)

print(f"      Feature matrix: {X.shape}")
print(f"      Positive samples: {y.sum()} ({100*y.mean():.1f}%)")

# ============================================================================
# STEP 4: Scale features and split data
# ============================================================================
print("\n[4/7] Scaling features and splitting data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"      Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# STEP 5: Train Random Forest
# ============================================================================
print("\n[5/7] Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    max_features=MAX_FEATURES,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced',
    verbose=1
)

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\n      Train accuracy: {train_score:.4f}")
print(f"      Test accuracy: {test_score:.4f}")

with open(CHECKPOINT_DIR / 'P4_improved_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

# ============================================================================
# STEP 6: Cross-validation
# ============================================================================
print("\n[6/7] Cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"      CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ============================================================================
# STEP 7: Generate per-patient recommendations
# Rank pathways by prediction score for each Crohn's patient
# ============================================================================
print("\n[7/7] Generating recommendations...")
y_pred_proba = model.predict_proba(X_scaled)[:, 1]
metadata['prediction_score'] = y_pred_proba

print(f"\n      Prediction stats:")
print(f"        Mean: {y_pred_proba.mean():.4f}")
print(f"        Std:  {y_pred_proba.std():.4f}")
print(f"        Min:  {y_pred_proba.min():.4f}")
print(f"        Max:  {y_pred_proba.max():.4f}")

recommendations = defaultdict(list)
crohns_data = metadata[metadata['is_crohns']].copy()

for patient_id in crohns_patients:
    patient_data = crohns_data[crohns_data['patient_id'] == patient_id]
    if len(patient_data) > 0:
        top_pathways = patient_data.nlargest(20, 'prediction_score')
        recommendations[patient_id] = top_pathways[
            ['pathway_id', 'prediction_score', 'pathway_activity', 'nes', 'fdr']
        ].to_dict('records')

# Rename pathway_id to pathway for downstream compatibility
for patient_id in recommendations:
    for rec in recommendations[patient_id]:
        rec['pathway'] = rec.pop('pathway_id')

with open(RESULTS_DIR / 'drug_patient_recommendations_improved.json', 'w') as f:
    json.dump(recommendations, f, indent=2)

print(f"      Saved recommendations for {len(recommendations)} patients")

# ============================================================================
# VISUALIZATION: Recommendation confidence per patient
# Panel B: Mean prediction score per patient
# Panel C: High vs medium confidence counts per patient
#   - Shows full picture beyond top 20 in patient reports
# ============================================================================
tissue_name = 'Colon'

patient_scores = {}
all_scores = []
for patient_id, recs in recommendations.items():
    scores = [rec['prediction_score'] for rec in recs]
    patient_scores[patient_id] = {
        'mean': np.mean(scores),
        'high_conf': sum(1 for s in scores if s > 0.7),
        'med_conf': sum(1 for s in scores if 0.5 <= s <= 0.7)
    }
    all_scores.extend(scores)

patient_ids = sorted(patient_scores.keys())
mean_scores = [patient_scores[pid]['mean'] for pid in patient_ids]
high_conf_counts = [patient_scores[pid]['high_conf'] for pid in patient_ids]
med_conf_counts = [patient_scores[pid]['med_conf'] for pid in patient_ids]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel B: Mean prediction score per patient
colors_patients = plt.cm.RdYlGn(np.interp(mean_scores, [min(mean_scores), max(mean_scores)], [0.2, 0.9]))
axes[0].bar(range(len(patient_ids)), mean_scores, color=colors_patients, edgecolor='black')
axes[0].set_xticks(range(len(patient_ids)))
axes[0].set_xticklabels([f'P{pid}' for pid in patient_ids], rotation=90, fontsize=8)
axes[0].axhline(0.7, color='red', linestyle='--', alpha=0.5, label='High conf')
axes[0].axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Med conf')
axes[0].set_ylabel('Mean Prediction Score', fontweight='bold')
axes[0].set_title('B) Mean Score per Patient', fontweight='bold')
axes[0].legend()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Panel C: High vs medium confidence counts per patient
x = np.arange(len(patient_ids))
width = 0.35
axes[1].bar(x - width/2, high_conf_counts, width, label='High (>0.7)', color='#2ECC71', alpha=0.8)
axes[1].bar(x + width/2, med_conf_counts, width, label='Med (0.5-0.7)', color='#F39C12', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'P{pid}' for pid in patient_ids], rotation=90, fontsize=8)
axes[1].set_ylabel('Count', fontweight='bold')
axes[1].set_title('C) Confidence Distribution per Patient', fontweight='bold')
axes[1].legend()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('P4 Recommendation Confidence - Colon', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / f'Recommendation_Confidence_{tissue_name}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n      Saved: Recommendation_Confidence_{tissue_name}.png")

# ============================================================================
# STEP 5 FIGURE: Model Training Performance
# ============================================================================
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

print("\n      Generating Step5 training performance figure...")

y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

prec_train, rec_train, _ = precision_recall_curve(y_train, y_train_proba)
prec_test, rec_test, _ = precision_recall_curve(y_test, y_test_proba)
pr_auc_train = auc(rec_train, prec_train)
pr_auc_test = auc(rec_test, prec_test)

cm = confusion_matrix(y_test, model.predict(X_test))
baseline_pr = y_test.mean()

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle('P4 Model Training Performance', fontsize=16, fontweight='bold')

# A) ROC Curve
axes[0,0].plot(fpr_train, tpr_train, color='#2E86AB', lw=2, label=f'Train (AUC = {auc_train:.3f})')
axes[0,0].plot(fpr_test, tpr_test, color='#E84855', lw=2, label=f'Test (AUC = {auc_test:.3f})')
axes[0,0].plot([0,1],[0,1], 'k--', lw=1.5, label='Random')
axes[0,0].set_xlabel('False Positive Rate', fontweight='bold')
axes[0,0].set_ylabel('True Positive Rate', fontweight='bold')
axes[0,0].set_title('A) ROC Curve', fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# B) Precision-Recall Curve
axes[0,1].plot(rec_train, prec_train, color='#2E86AB', lw=2, label=f'Train (AUC = {pr_auc_train:.3f})')
axes[0,1].plot(rec_test, prec_test, color='#E84855', lw=2, label=f'Test (AUC = {pr_auc_test:.3f})')
axes[0,1].axhline(y=baseline_pr, color='gray', linestyle='--', lw=1.5, label=f'Baseline ({baseline_pr:.3f})')
axes[0,1].set_xlabel('Recall', fontweight='bold')
axes[0,1].set_ylabel('Precision', fontweight='bold')
axes[0,1].set_title('B) Precision-Recall Curve', fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# C) Confusion Matrix
im = axes[0,2].imshow(cm / cm.sum(axis=1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
axes[0,2].set_xticks([0,1]); axes[0,2].set_yticks([0,1])
axes[0,2].set_xticklabels(['Negative','Positive'])
axes[0,2].set_yticklabels(['Negative','Positive'])
axes[0,2].set_xlabel('Predicted', fontweight='bold')
axes[0,2].set_ylabel('Actual', fontweight='bold')
axes[0,2].set_title('C) Test Set Confusion Matrix', fontweight='bold')
for i in range(2):
    for j in range(2):
        axes[0,2].text(j, i, f'{cm[i,j]}\n({cm[i,j]/cm[i].sum()*100:.1f}%)',
                      ha='center', va='center', fontsize=11, fontweight='bold',
                      color='white' if cm[i,j]/cm.sum() > 0.3 else 'black')

# D) Prediction Score Distribution
axes[1,0].hist(y_pred_proba[y==0], bins=50, alpha=0.6, color='#2E86AB', label='Train Negative', density=False)
axes[1,0].hist(y_pred_proba[y==1], bins=50, alpha=0.6, color='#E84855', label='Train Positive', density=False)
axes[1,0].axvline(x=0.5, color='black', linestyle='--', lw=2)
axes[1,0].set_xlabel('Prediction Score', fontweight='bold')
axes[1,0].set_ylabel('Frequency', fontweight='bold')
axes[1,0].set_title('D) Prediction Score Distribution', fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# E) Feature Importance
feature_names = [f'emb_{i}' for i in range(20)] + [
    'pathway_activity', 'pop_signature', 'cosine_sim', 'nes', 'fdr', 'rel_activity'
]
importances = model.feature_importances_
sorted_idx = np.argsort(importances)
top_idx = sorted_idx[-15:]
axes[1,1].barh(range(15), importances[top_idx], color='#85C1E9')
axes[1,1].set_yticks(range(15))
axes[1,1].set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
axes[1,1].set_xlabel('Feature Importance', fontweight='bold')
axes[1,1].set_title('E) Top 15 Feature Importances', fontweight='bold')
axes[1,1].grid(axis='x', alpha=0.3)

# F) Train vs Test Performance
metrics = ['Accuracy', 'ROC AUC', 'PR AUC']
train_vals = [train_score, auc_train, pr_auc_train]
test_vals = [test_score, auc_test, pr_auc_test]
x_pos = np.arange(len(metrics))
width = 0.35
axes[2,0].bar(x_pos - width/2, train_vals, width, label='Train', color='#2E86AB', alpha=0.8)
axes[2,0].bar(x_pos + width/2, test_vals, width, label='Test', color='#E84855', alpha=0.8)
for i, (tv, v) in enumerate(zip(train_vals, test_vals)):
    axes[2,0].text(i - width/2, tv + 0.01, f'{tv:.3f}', ha='center', fontsize=9, fontweight='bold')
    axes[2,0].text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[2,0].set_xticks(x_pos)
axes[2,0].set_xticklabels(metrics)
axes[2,0].set_ylabel('Score', fontweight='bold')
axes[2,0].set_title('F) Train vs Test Performance', fontweight='bold')
axes[2,0].legend()
axes[2,0].set_ylim([0, 1.1])
axes[2,0].grid(axis='y', alpha=0.3)

# G) Prediction Statistics Table
pred_stats = {
    'Metric': ['Mean Score', 'Std Dev', 'Min Score', 'Max Score', 'Median'],
    'Train': [f'{y_pred_proba[y==1].mean():.4f}', f'{y_pred_proba.std():.4f}',
              f'{y_pred_proba.min():.4f}', f'{y_pred_proba.max():.4f}',
              f'{np.median(y_pred_proba):.4f}'],
    'Test': [f'{y_test_proba.mean():.4f}', f'{y_test_proba.std():.4f}',
             f'{y_test_proba.min():.4f}', f'{y_test_proba.max():.4f}',
             f'{np.median(y_test_proba):.4f}']
}
axes[1,2].axis('off')
table = axes[1,2].table(
    cellText=list(zip(pred_stats['Metric'], pred_stats['Train'], pred_stats['Test'])),
    colLabels=['Metric', 'Train', 'Test'],
    cellLoc='center', loc='center',
    colColours=['#764ba2','#764ba2','#764ba2']
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_text_props(color='white', fontweight='bold')
axes[1,2].set_title('G) Prediction Statistics', fontweight='bold')

# H) Model Configuration Table
config_data = [
    ['n_estimators', str(N_ESTIMATORS)],
    ['max_depth', str(MAX_DEPTH)],
    ['min_samples_split', str(MIN_SAMPLES_SPLIT)],
    ['min_samples_leaf', str(MIN_SAMPLES_LEAF)],
    ['max_features', str(MAX_FEATURES)],
    ['class_weight', 'balanced']
]
axes[2,2].axis('off')
config_table = axes[2,2].table(
    cellText=config_data,
    colLabels=['Hyperparameter', 'Value'],
    cellLoc='center', loc='center',
    colColours=['#764ba2','#764ba2']
)
config_table.auto_set_font_size(False)
config_table.set_fontsize(10)
config_table.scale(1.2, 1.8)
for (r, c), cell in config_table.get_celld().items():
    if r == 0:
        cell.set_text_props(color='white', fontweight='bold')
axes[2,2].set_title('H) Model Configuration', fontweight='bold')

axes[2,1].axis('off')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'Step5_Training_Performance_Colon.png', dpi=300, bbox_inches='tight')
plt.close()
print("      Saved: Step5_Training_Performance_Colon.png")

# ============================================================================
# STEP 6 FIGURE: Cross-Validation Results
# ============================================================================
print("\n      Generating Step6 cross-validation figure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('P4 Cross-Validation Results', fontsize=14, fontweight='bold')

# A) Boxplot of CV scores
axes[0,0].boxplot(cv_scores, patch_artist=True,
                  boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                  medianprops=dict(color='orange', linewidth=2))
axes[0,0].scatter([1]*len(cv_scores), cv_scores, color='#E84855', s=50, zorder=5)
axes[0,0].axhline(y=cv_scores.mean(), color='#2E86AB', linestyle='--', lw=2,
                  label=f'Mean: {cv_scores.mean():.4f}')
axes[0,0].set_xticklabels(['CV AUC Scores'])
axes[0,0].set_ylabel('AUC Score', fontweight='bold')
axes[0,0].set_title('A) Cross-Validation AUC Scores', fontweight='bold')
axes[0,0].legend()
axes[0,0].grid(axis='y', alpha=0.3)

# B) Performance per fold
colors_cv = ['#2E86AB' if s >= cv_scores.mean() else '#E84855' for s in cv_scores]
bars = axes[0,1].bar(range(1, 6), cv_scores, color=colors_cv, alpha=0.85,
                     edgecolor='black', linewidth=1.5, width=0.6)
axes[0,1].axhline(y=cv_scores.mean(), color='red', linestyle='--', lw=2,
                  label=f'Mean: {cv_scores.mean():.3f}')
axes[0,1].axhline(y=cv_scores.mean() + cv_scores.std(), color='gray',
                  linestyle=':', lw=2, label=f'±1 Std: {cv_scores.std():.3f}')
axes[0,1].axhline(y=cv_scores.mean() - cv_scores.std(), color='gray',
                  linestyle=':', lw=2)
for bar, score in zip(bars, cv_scores):
    axes[0,1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0,1].set_xlabel('Fold Number', fontweight='bold')
axes[0,1].set_ylabel('AUC Score', fontweight='bold')
axes[0,1].set_title('B) Performance per Fold', fontweight='bold')
axes[0,1].legend(loc='lower right')
axes[0,1].set_ylim([0.5, 1.0])
axes[0,1].set_xticks(range(1, 6))
axes[0,1].grid(axis='y', alpha=0.3)

# C) Overall Performance Summary
summary_metrics = ['Train\nAccuracy', 'Test\nAccuracy', 'Mean\nCV AUC']
summary_vals = [train_score, test_score, cv_scores.mean()]
colors_summary = ['#2E86AB', '#E84855', '#2ECC71']
bars_s = axes[1,0].bar(summary_metrics, summary_vals, color=colors_summary, alpha=0.8,
                        edgecolor='black')
for bar, val in zip(bars_s, summary_vals):
    axes[1,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[1,0].set_ylabel('Score', fontweight='bold')
axes[1,0].set_title('C) Overall Performance Summary', fontweight='bold')
axes[1,0].set_ylim([0, 1.1])
axes[1,0].grid(axis='y', alpha=0.3)

# D) Robustness Metrics Table
train_test_gap = train_score - test_score
generalization = 'Good' if train_test_gap < 0.2 else 'Moderate' if train_test_gap < 0.3 else 'Poor'
robustness_data = [
    ['CV Mean AUC', f'{cv_scores.mean():.4f}'],
    ['CV Std Dev', f'{cv_scores.std():.4f}'],
    ['CV Min', f'{cv_scores.min():.4f}'],
    ['CV Max', f'{cv_scores.max():.4f}'],
    ['CV Range', f'{cv_scores.max()-cv_scores.min():.4f}'],
    ['Train-Test Gap', f'{train_test_gap:.4f}'],
    ['Generalization', generalization]
]
axes[1,1].axis('off')
rob_table = axes[1,1].table(
    cellText=robustness_data,
    colLabels=['Metric', 'Value'],
    cellLoc='center', loc='center',
    colColours=['#764ba2','#764ba2']
)
rob_table.auto_set_font_size(False)
rob_table.set_fontsize(10)
rob_table.scale(1.2, 1.8)
for (r, c), cell in rob_table.get_celld().items():
    if r == 0:
        cell.set_text_props(color='white', fontweight='bold')
axes[1,1].set_title('D) Model Robustness Metrics', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'Step6_CrossValidation_Colon.png', dpi=300, bbox_inches='tight')
plt.close()
print("      Saved: Step6_CrossValidation_Colon.png")

# Feature importance saved for P5
feature_names = [f'emb_{i}' for i in range(20)] + [
    'pathway_activity', 'population_signature', 'cosine_similarity',
    'nes', 'fdr', 'relative_activity'
]
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.to_csv(RESULTS_DIR / 'feature_importance_improved.csv', index=False)

print("\n" + "="*80)
print("P4 COMPLETE - COLON")
print("="*80)
print(f"Model:          Random Forest (n_estimators={N_ESTIMATORS})")
print(f"Train accuracy: {train_score:.3f}")
print(f"Test accuracy:  {test_score:.3f}")
print(f"CV AUC:         {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Recommendations:{len(recommendations)} patients")
print("="*80)