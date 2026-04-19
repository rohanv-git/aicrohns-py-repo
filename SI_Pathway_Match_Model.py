"""
P4: Drug-Patient Matching with Random Forest
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
CHECKPOINT_DIR = Path('checkpoints')
RESULTS_DIR = Path('results/P4_results')
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
print("P4: DRUG-PATIENT MATCHING")
print("="*80)

# ============================================================================
# STEP 1: Load patient metadata from small intestine
# ============================================================================
print("\n[1/7] Loading patient information...")
adata_full = sc.read_h5ad(ADATA_FILE)
adata = adata_full[adata_full.obs['organ__ontology_label'] == 'small intestine'].copy()

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
tissue_name = 'SmallIntestine'

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

plt.suptitle('P4 Recommendation Confidence', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / f'Recommendation_Confidence_{tissue_name}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n      Saved: Recommendation_Confidence_{tissue_name}.png")

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
print("P4 COMPLETE")
print("="*80)
print(f"Model:          Random Forest (n_estimators={N_ESTIMATORS})")
print(f"Train accuracy: {train_score:.3f}")
print(f"Test accuracy:  {test_score:.3f}")
print(f"CV AUC:         {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Recommendations:{len(recommendations)} patients")
print("="*80)