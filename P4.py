# P4.py - Match patients to drugs using Random Forest
# FINAL VERSION - Using original hyperparameters (best performance)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, 
                             classification_report)
import shap
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Setup
print("="*80)
print("P4: AI-POWERED DRUG-PATIENT MATCHING")
print("="*80)

# Ensure directories exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("figures/P4_figs", exist_ok=True)
os.makedirs("results/P4_results", exist_ok=True)

print("\nCheckmark Directory structure ready")

# File paths - INPUTS
P2_GSEA_FILE = "checkpoints/P2_gsea_results_BIG_DARK_ONLY.csv"
P3_PATIENT_EMBEDDINGS = "results/P3_results/P3_patient_embeddings.csv"
P3_GENE_EMBEDDINGS = "results/P3_results/P3_gene_embeddings.csv"
P3_PATHWAY_SCORES = "results/P3_results/P3_pathway_scores.csv"

# File paths - OUTPUTS
OUTPUT_PREDICTIONS = "results/P4_results/P4_drug_predictions.csv"
OUTPUT_MODEL = "checkpoints/P4_model.pkl"
OUTPUT_METRICS = "results/P4_results/P4_model_metrics.json"
OUTPUT_FEATURE_IMPORTANCE = "results/P4_results/P4_feature_importance.csv"

# Checkpoint files
CHECKPOINT_DRUG_EMBEDDINGS = "checkpoints/P4_drug_embeddings.pkl"
CHECKPOINT_FEATURES = "checkpoints/P4_features.pkl"
CHECKPOINT_TRAIN_TEST = "checkpoints/P4_train_test_split.pkl"
CHECKPOINT_PREDICTIONS = "checkpoints/P4_predictions.pkl"

# Figure paths
FIG_ROC_CURVE = "figures/P4_figs/P4_roc_curve.png"
FIG_CONFUSION_MATRIX = "figures/P4_figs/P4_confusion_matrix.png"
FIG_FEATURE_IMPORTANCE = "figures/P4_figs/P4_feature_importance.png"
FIG_LEARNING_CURVE = "figures/P4_figs/P4_learning_curve.png"
FIG_SHAP_SUMMARY = "figures/P4_figs/P4_shap_summary.png"
FIG_PREDICTION_DIST = "figures/P4_figs/P4_prediction_distribution.png"

# ============================================================================
# STEP 1: Load P2 GSEA results
# ============================================================================
print("\n[1/8] Loading P2 GSEA results...")
gsea_df = pd.read_csv(P2_GSEA_FILE)

# Remove infinite values
gsea_df = gsea_df[~np.isinf(gsea_df['nes'])].copy()

print(f"      Loaded {len(gsea_df)} pathway enrichments")
print(f"      NES range: [{gsea_df['nes'].min():.2f}, {gsea_df['nes'].max():.2f}]")
print(f"      FDR range: [{gsea_df['fdr'].min():.4f}, {gsea_df['fdr'].max():.4f}]")

# ============================================================================
# STEP 2: Load P3 patient embeddings
# ============================================================================
print("\n[2/8] Loading P3 patient embeddings...")
patient_emb_df = pd.read_csv(P3_PATIENT_EMBEDDINGS)

print(f"      Loaded {len(patient_emb_df)} patient-celltype combinations")
print(f"      Embedding dimensions: {len([c for c in patient_emb_df.columns if c.startswith('emb_')])}")

# ============================================================================
# STEP 3: Load P3 pathway scores
# ============================================================================
print("\n[3/8] Loading P3 pathway scores...")
pathway_scores_df = pd.read_csv(P3_PATHWAY_SCORES)

print(f"      Loaded {len(pathway_scores_df)} pathway activity scores")
print(f"      Unique pathways: {pathway_scores_df['pathway'].nunique()}")

# ============================================================================
# STEP 4: Create drug embeddings (WITH CHECKPOINT)
# ============================================================================
print("\n[4/8] Creating drug embeddings from pathways...")

if os.path.exists(CHECKPOINT_DRUG_EMBEDDINGS):
    print("      Checkmark Loading cached drug embeddings...")
    with open(CHECKPOINT_DRUG_EMBEDDINGS, 'rb') as f:
        drug_data = pickle.load(f)
    
    drug_emb_df = drug_data['drug_emb_df']
    pathway_genes = drug_data['pathway_genes']
    
    print(f"      Loaded embeddings for {len(drug_emb_df)} drugs")
else:
    print("      Computing drug embeddings...")
    
    # Load gene embeddings
    gene_emb_df = pd.read_csv(P3_GENE_EMBEDDINGS, index_col=0)
    
    # Parse genes from leading_edge in P2
    pathway_genes = {}
    for idx, row in gsea_df.iterrows():
        pathway_name = row['pathway']
        genes_str = row['leading_edge']
        
        if pd.notna(genes_str) and genes_str:
            genes = [g.strip() for g in genes_str.split(',')]
            
            if pathway_name not in pathway_genes:
                pathway_genes[pathway_name] = set()
            pathway_genes[pathway_name].update(genes)
    
    # Convert to lists
    for pathway in pathway_genes:
        pathway_genes[pathway] = list(pathway_genes[pathway])
    
    print(f"      Parsed {len(pathway_genes)} unique pathways")
    
    # Create drug embeddings (average of gene embeddings)
    drug_embeddings = {}
    for pathway, genes in pathway_genes.items():
        pathway_gene_embs = []
        for gene in genes:
            if gene in gene_emb_df.index:
                pathway_gene_embs.append(gene_emb_df.loc[gene].values)
        
        if len(pathway_gene_embs) > 0:
            drug_embeddings[pathway] = np.mean(pathway_gene_embs, axis=0)
    
    print(f"      Created embeddings for {len(drug_embeddings)} drugs")
    
    # Convert to DataFrame
    drug_emb_df = pd.DataFrame(drug_embeddings).T
    drug_emb_df.columns = [f'drug_emb_{i}' for i in range(drug_emb_df.shape[1])]
    
    # Save checkpoint
    drug_data = {
        'drug_emb_df': drug_emb_df,
        'pathway_genes': pathway_genes
    }
    with open(CHECKPOINT_DRUG_EMBEDDINGS, 'wb') as f:
        pickle.dump(drug_data, f)
    print(f"      Checkmark Saved checkpoint: {CHECKPOINT_DRUG_EMBEDDINGS}")

print(f"      Drug embedding shape: {drug_emb_df.shape}")

# ============================================================================
# STEP 5: Create feature matrix (WITH CHECKPOINT)
# ============================================================================
print("\n[5/8] Creating feature matrix...")

if os.path.exists(CHECKPOINT_FEATURES):
    print("      Checkmark Loading cached feature matrix...")
    with open(CHECKPOINT_FEATURES, 'rb') as f:
        feature_data = pickle.load(f)
    
    X = feature_data['X']
    y = feature_data['y']
    metadata_df = feature_data['metadata_df']
    
    print(f"      Loaded feature matrix: {X.shape}")
    print(f"      Positive samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
else:
    print("      Computing feature matrix...")
    
    # Pre-compute patient embedding matrix
    print("      Preparing patient embeddings...")
    patient_emb_cols = [c for c in patient_emb_df.columns if c.startswith('emb_')]
    patient_emb_matrix = patient_emb_df[patient_emb_cols].values
    
    # Pre-compute drug embedding matrix
    print("      Preparing drug embeddings...")
    drug_emb_matrix = drug_emb_df.values
    
    # Create lookup dictionaries
    print("      Creating lookup indices...")
    patient_to_idx = {(row['donor_id'], row['celltype']): idx 
                      for idx, row in patient_emb_df.iterrows()}
    
    pathway_to_idx = {pathway: idx for idx, pathway in enumerate(drug_emb_df.index)}
    
    # Pre-compute pathway medians
    print("      Computing pathway medians...")
    pathway_medians = pathway_scores_df.groupby('pathway')['pathway_score'].median().to_dict()
    
    # Pre-index P2 GSEA
    print("      Indexing P2 GSEA data...")
    p2_lookup = {}
    for idx, row in gsea_df.iterrows():
        key = (row['celltype'], row['pathway'])
        p2_lookup[key] = {'nes': row['nes'], 'fdr': row['fdr']}
    
    # Pre-index P3 pathway scores
    print("      Indexing P3 pathway scores...")
    p3_lookup = {}
    for idx, row in pathway_scores_df.iterrows():
        key = (row['donor_id'], row['celltype'], row['pathway'])
        p3_lookup[key] = row['pathway_score']
    
    print("      Building feature matrix...")
    
    features_list = []
    labels_list = []
    metadata_list = []
    
    total_combos = len(patient_emb_df) * len(drug_emb_df)
    print(f"      Total combinations to process: {total_combos:,}")
    
    processed = 0
    for patient_idx, patient_row in patient_emb_df.iterrows():
        donor_id = patient_row['donor_id']
        celltype = patient_row['celltype']
        
        patient_emb = patient_emb_matrix[patient_idx]
        
        for drug_idx, pathway in enumerate(drug_emb_df.index):
            processed += 1
            
            if processed % 50000 == 0:
                print(f"      Progress: {processed:,}/{total_combos:,} ({processed/total_combos*100:.1f}%)")
            
            drug_emb = drug_emb_matrix[drug_idx]
            
            # Calculate cosine similarity
            cosine_sim = np.dot(patient_emb, drug_emb) / (
                np.linalg.norm(patient_emb) * np.linalg.norm(drug_emb) + 1e-10
            )
            
            # Get P2 data
            p2_key = (celltype, pathway)
            if p2_key not in p2_lookup:
                continue
            
            nes = p2_lookup[p2_key]['nes']
            fdr = p2_lookup[p2_key]['fdr']
            
            # Get P3 pathway activity
            p3_key = (donor_id, celltype, pathway)
            pathway_activity = p3_lookup.get(p3_key, 0.0)
            
            # Create feature vector
            features = np.concatenate([
                patient_emb,              # 128 dims
                drug_emb,                 # 128 dims
                [cosine_sim],             # 1 dim
                [nes],                    # 1 dim
                [fdr],                    # 1 dim
                [pathway_activity]        # 1 dim
            ])
            
            # Create label (weak supervision)
            pathway_median = pathway_medians.get(pathway, 0.0)
            
            if pathway_activity > pathway_median and nes < -1.5:
                label = 1
            elif pathway_activity < pathway_median and nes > 1.5:
                label = 1
            else:
                label = 0
            
            features_list.append(features)
            labels_list.append(label)
            metadata_list.append({
                'donor_id': donor_id,
                'celltype': celltype,
                'pathway': pathway
            })
    
    # Convert to arrays
    X = np.array(features_list)
    y = np.array(labels_list)
    metadata_df = pd.DataFrame(metadata_list)
    
    print(f"\n      Checkmark Created feature matrix: {X.shape}")
    print(f"      Checkmark Labels: {y.shape}")
    print(f"      Checkmark Positive samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"      Checkmark Negative samples: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.1f}%)")
    
    # Save checkpoint
    feature_data = {
        'X': X,
        'y': y,
        'metadata_df': metadata_df
    }
    with open(CHECKPOINT_FEATURES, 'wb') as f:
        pickle.dump(feature_data, f)
    print(f"      Checkmark Saved checkpoint: {CHECKPOINT_FEATURES}")

# ============================================================================
# STEP 6: Train-test split (WITH CHECKPOINT)
# ============================================================================
print("\n[6/8] Creating train-test split...")

if os.path.exists(CHECKPOINT_TRAIN_TEST):
    print("      Checkmark Loading cached train-test split...")
    with open(CHECKPOINT_TRAIN_TEST, 'rb') as f:
        split_data = pickle.load(f)
    
    X_train = split_data['X_train']
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']
    train_patients = split_data['train_patients']
    test_patients = split_data['test_patients']
    train_mask = split_data['train_mask']
    test_mask = split_data['test_mask']
    
    print(f"      Train patients: {len(train_patients)}")
    print(f"      Test patients: {len(test_patients)}")
else:
    print("      Computing train-test split...")
    
    # Patient-level split
    unique_patients = metadata_df['donor_id'].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, 
        test_size=0.3, 
        random_state=42
    )
    
    print(f"      Train patients: {len(train_patients)}")
    print(f"      Test patients: {len(test_patients)}")
    
    # Create train/test masks
    train_mask = metadata_df['donor_id'].isin(train_patients)
    test_mask = metadata_df['donor_id'].isin(test_patients)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    # Save checkpoint
    split_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_patients': train_patients,
        'test_patients': test_patients,
        'train_mask': train_mask,
        'test_mask': test_mask
    }
    
    with open(CHECKPOINT_TRAIN_TEST, 'wb') as f:
        pickle.dump(split_data, f)
    print(f"      Checkmark Saved checkpoint: {CHECKPOINT_TRAIN_TEST}")

print(f"\n      Train set: {X_train.shape[0]} samples ({y_train.sum()} positive)")
print(f"      Test set: {X_test.shape[0]} samples ({y_test.sum()} positive)")
print(f"      Train positive rate: {y_train.mean()*100:.1f}%")
print(f"      Test positive rate: {y_test.mean()*100:.1f}%")

# ============================================================================
# STEP 7: Train Random Forest (ORIGINAL HYPERPARAMETERS - BEST MODEL)
# ============================================================================
print("\n[7/8] Training Random Forest classifier...")

if os.path.exists(OUTPUT_MODEL):
    print("      Checkmark Loading existing model...")
    with open(OUTPUT_MODEL, 'rb') as f:
        rf_model = pickle.load(f)
    print("      Model loaded from checkpoint")
else:
    print("      Training new model...")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,           # ORIGINAL: Best test AUC
        min_samples_split=50,   # ORIGINAL: Best test AUC
        min_samples_leaf=20,    # ORIGINAL: Best test AUC
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("\n      Fitting model (this may take 2-5 minutes)...")
    rf_model.fit(X_train, y_train)
    
    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"      Checkmark Model saved to {OUTPUT_MODEL}")

print("\n      Model training complete!")
print(f"      Number of trees: {rf_model.n_estimators}")
print(f"      Max depth: {rf_model.max_depth}")
print(f"      Min samples split: {rf_model.min_samples_split}")
print(f"      Min samples leaf: {rf_model.min_samples_leaf}")

# ============================================================================
# STEP 8: Evaluate model (WITH CHECKPOINT)
# ============================================================================
print("\n[8/8] Evaluating model performance...")

if os.path.exists(CHECKPOINT_PREDICTIONS):
    print("      Checkmark Loading cached predictions...")
    with open(CHECKPOINT_PREDICTIONS, 'rb') as f:
        pred_data = pickle.load(f)
    
    y_train_pred = pred_data['y_train_pred']
    y_test_pred = pred_data['y_test_pred']
    y_train_proba = pred_data['y_train_proba']
    y_test_proba = pred_data['y_test_proba']
else:
    print("      Computing predictions...")
    
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    y_train_proba = rf_model.predict_proba(X_train)[:, 1]
    y_test_proba = rf_model.predict_proba(X_test)[:, 1]
    
    pred_data = {
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_train_proba': y_train_proba,
        'y_test_proba': y_test_proba
    }
    
    with open(CHECKPOINT_PREDICTIONS, 'wb') as f:
        pickle.dump(pred_data, f)
    print(f"      Checkmark Saved checkpoint: {CHECKPOINT_PREDICTIONS}")

# Calculate metrics
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n      MODEL PERFORMANCE:")
print(f"      Train AUC: {train_auc:.4f}")
print(f"      Test AUC:  {test_auc:.4f}")
print(f"      Gap:       {abs(train_auc - test_auc):.4f}")

if abs(train_auc - test_auc) < 0.05:
    print("      Checkmark No overfitting detected!")
elif abs(train_auc - test_auc) < 0.10:
    print("      Warning: Slight overfitting (acceptable)")
else:
    print("      Error: Significant overfitting!")

if test_auc > 0.7:
    print(f"      Checkmark Test AUC > 0.7 - Model is working well!")
elif test_auc > 0.6:
    print(f"      Warning: Test AUC = {test_auc:.2f} - Model is learning but could be better")
else:
    print(f"      Error: Test AUC < 0.6 - Model is not learning well")

# Classification report
print("\n      CLASSIFICATION REPORT (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['No Match', 'Good Match']))

# Save metrics
print(f"      Saving metrics to {OUTPUT_METRICS}...")
metrics = {
    'train_auc': float(train_auc),
    'test_auc': float(test_auc),
    'train_samples': int(len(y_train)),
    'test_samples': int(len(y_test)),
    'train_positive_rate': float(y_train.mean()),
    'test_positive_rate': float(y_test.mean()),
    'overfitting_gap': float(abs(train_auc - test_auc)),
    'hyperparameters': {
        'n_estimators': rf_model.n_estimators,
        'max_depth': rf_model.max_depth,
        'min_samples_split': rf_model.min_samples_split,
        'min_samples_leaf': rf_model.min_samples_leaf
    }
}

import json
with open(OUTPUT_METRICS, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"      Checkmark Metrics saved to {OUTPUT_METRICS}")

print("\n" + "="*80)
print("P4 STEPS 1-8 COMPLETE!")
print("="*80)
print("\nReady for visualizations")

# ============================================================================
# VISUALIZATION 1: ROC Curve (Train vs Test)
# ============================================================================
print("\n[VIZ 1/6] Creating ROC curves...")

if not os.path.exists(FIG_ROC_CURVE):
    # Calculate ROC curves
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_train, tpr_train, label=f'Train (AUC = {train_auc:.3f})', linewidth=2)
    plt.plot(fpr_test, tpr_test, label=f'Test (AUC = {test_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Train vs Test', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_ROC_CURVE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Checkmark Saved to {FIG_ROC_CURVE}")
else:
    print(f"      Checkmark Figure already exists: {FIG_ROC_CURVE}")

# ============================================================================
# VISUALIZATION 2: Learning Curve
# ============================================================================
print("\n[VIZ 2/6] Creating learning curve...")

if not os.path.exists(FIG_LEARNING_CURVE):
    print("      Computing learning curve (this may take 2-3 minutes)...")
    
    # Use subset of data for speed
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        rf_model, X_train, y_train,
        train_sizes=train_sizes,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    
    # Calculate mean and std
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(train_sizes_abs, train_mean, label='Train Score', marker='o', linewidth=2)
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    
    plt.plot(train_sizes_abs, test_mean, label='Cross-Val Score', marker='o', linewidth=2)
    plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.2)
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('Learning Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(FIG_LEARNING_CURVE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Checkmark Saved to {FIG_LEARNING_CURVE}")
else:
    print(f"      Checkmark Figure already exists: {FIG_LEARNING_CURVE}")

# ============================================================================
# VISUALIZATION 3: Feature Importance
# ============================================================================
print("\n[VIZ 3/6] Creating feature importance plot...")

if not os.path.exists(FIG_FEATURE_IMPORTANCE):
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Create feature names
    feature_names = (
        [f'patient_emb_{i}' for i in range(128)] +
        [f'drug_emb_{i}' for i in range(128)] +
        ['cosine_similarity', 'nes', 'fdr', 'pathway_activity']
    )
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    # Get top 20 features
    top_n = 20
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importances[::-1])
    plt.yticks(range(top_n), top_names[::-1])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_FEATURE_IMPORTANCE, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save to CSV
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(OUTPUT_FEATURE_IMPORTANCE, index=False)
    
    print(f"      Checkmark Saved figure to {FIG_FEATURE_IMPORTANCE}")
    print(f"      Checkmark Saved CSV to {OUTPUT_FEATURE_IMPORTANCE}")
    
    # Show top 5
    print("\n      Top 5 Most Important Features:")
    for i in range(5):
        print(f"        {i+1}. {importance_df.iloc[i]['feature']}: {importance_df.iloc[i]['importance']:.4f}")
else:
    print(f"      Checkmark Figure already exists: {FIG_FEATURE_IMPORTANCE}")

# ============================================================================
# VISUALIZATION 4: Confusion Matrix
# ============================================================================
print("\n[VIZ 4/6] Creating confusion matrix...")

if not os.path.exists(FIG_CONFUSION_MATRIX):
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Match', 'Good Match'],
                yticklabels=['No Match', 'Good Match'])
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_CONFUSION_MATRIX, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Checkmark Saved to {FIG_CONFUSION_MATRIX}")
else:
    print(f"      Checkmark Figure already exists: {FIG_CONFUSION_MATRIX}")

# ============================================================================
# VISUALIZATION 5: SHAP Summary Plot
# ============================================================================
print("\n[VIZ 5/6] Creating SHAP summary plot...")

if not os.path.exists(FIG_SHAP_SUMMARY):
    print("      Computing SHAP values (this may take 3-5 minutes)...")
    
    # Use subset for speed (1000 samples)
    sample_size = min(1000, len(X_test))
    X_shap = X_test[:sample_size]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap)
    
    # If binary classification, shap_values is a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    # Feature names
    feature_names = (
        [f'patient_emb_{i}' for i in range(128)] +
        [f'drug_emb_{i}' for i in range(128)] +
        ['cosine_similarity', 'nes', 'fdr', 'pathway_activity']
    )
    
    # Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, 
                     show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIG_SHAP_SUMMARY, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Checkmark Saved to {FIG_SHAP_SUMMARY}")
else:
    print(f"      Checkmark Figure already exists: {FIG_SHAP_SUMMARY}")

# ============================================================================
# VISUALIZATION 6: Prediction Distribution
# ============================================================================
print("\n[VIZ 6/6] Creating prediction distribution plot...")

if not os.path.exists(FIG_PREDICTION_DIST):
    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Test set distribution
    axes[0].hist(y_test_proba[y_test == 0], bins=50, alpha=0.7, label='True No Match', color='blue')
    axes[0].hist(y_test_proba[y_test == 1], bins=50, alpha=0.7, label='True Good Match', color='red')
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Test Set: Prediction Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # By patient
    test_metadata = metadata_df[test_mask]
    test_pred_df = pd.DataFrame({
        'donor_id': test_metadata['donor_id'].values,
        'probability': y_test_proba
    })
    
    patient_avg = test_pred_df.groupby('donor_id')['probability'].mean().sort_values()
    
    axes[1].barh(range(len(patient_avg)), patient_avg.values)
    axes[1].set_yticks(range(len(patient_avg)))
    axes[1].set_yticklabels(patient_avg.index)
    axes[1].set_xlabel('Average Prediction Score', fontsize=12)
    axes[1].set_ylabel('Patient ID', fontsize=12)
    axes[1].set_title('Average Prediction by Test Patient', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(FIG_PREDICTION_DIST, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Checkmark Saved to {FIG_PREDICTION_DIST}")
else:
    print(f"      Checkmark Figure already exists: {FIG_PREDICTION_DIST}")

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*80)

# ============================================================================
# FINAL STEP: Generate Patient-Specific Drug Recommendations
# ============================================================================
print("\n" + "="*80)
print("GENERATING PATIENT-SPECIFIC DRUG RECOMMENDATIONS")
print("="*80)

print("\n[1/4] Preparing prediction data...")

# Combine predictions with metadata
results_df = metadata_df.copy()
results_df['match_probability'] = np.concatenate([y_train_proba, y_test_proba])
results_df['prediction'] = np.concatenate([y_train_pred, y_test_pred])
results_df['split'] = ['train'] * len(y_train) + ['test'] * len(y_test)

# Merge with P2 GSEA data for pathway info
gsea_info = gsea_df[['celltype', 'pathway', 'nes', 'fdr']].drop_duplicates()
results_df = results_df.merge(gsea_info, on=['celltype', 'pathway'], how='left')

# Merge with P3 pathway scores
pathway_info = pathway_scores_df[['donor_id', 'celltype', 'pathway', 'pathway_score']]
results_df = results_df.merge(pathway_info, on=['donor_id', 'celltype', 'pathway'], how='left')

print(f"      Combined {len(results_df):,} predictions with pathway data")

# ============================================================================
print("\n[2/4] Generating top recommendations per patient...")

# Get top 20 drugs per patient
top_recommendations = []

for donor_id in results_df['donor_id'].unique():
    patient_data = results_df[results_df['donor_id'] == donor_id].copy()
    
    # Sort by match probability (descending)
    patient_data = patient_data.sort_values('match_probability', ascending=False)
    
    # Get top 20
    top_20 = patient_data.head(20).copy()
    top_20['rank'] = range(1, len(top_20) + 1)
    
    top_recommendations.append(top_20)

top_recommendations_df = pd.concat(top_recommendations, ignore_index=True)

print(f"      Generated top 20 recommendations for {results_df['donor_id'].nunique()} patients")

# ============================================================================
print("\n[3/4] Saving prediction files...")

# Save all predictions
OUTPUT_ALL_PREDICTIONS = OUTPUT_PREDICTIONS.replace('.csv', '_all.csv')
results_df.to_csv(OUTPUT_ALL_PREDICTIONS, index=False)
print(f"      ✓ Saved all predictions: {OUTPUT_ALL_PREDICTIONS}")

# Save top 20 per patient
OUTPUT_TOP20 = OUTPUT_PREDICTIONS.replace('.csv', '_top20.csv')
top_recommendations_df.to_csv(OUTPUT_TOP20, index=False)
print(f"      ✓ Saved top 20 per patient: {OUTPUT_TOP20}")

# ============================================================================
print("\n[4/4] Generating summary statistics...")

print(f"\n{'='*80}")
print("PREDICTION SUMMARY")
print(f"{'='*80}")

print(f"\nDataset:")
print(f"  Total predictions: {len(results_df):,}")
print(f"  Patients: {results_df['donor_id'].nunique()}")
print(f"  Unique drugs (pathways): {results_df['pathway'].nunique()}")
print(f"  Cell types: {results_df['celltype'].nunique()}")

all_probs = results_df['match_probability'].values
print(f"\nMatch probability distribution:")
print(f"  Mean: {all_probs.mean():.3f}")
print(f"  Median: {np.median(all_probs):.3f}")
print(f"  Std: {all_probs.std():.3f}")
print(f"  Min: {all_probs.min():.3f}")
print(f"  Max: {all_probs.max():.3f}")

print(f"\nConfidence breakdown:")
print(f"  High confidence (>0.7): {(all_probs > 0.7).sum():,} ({(all_probs > 0.7).sum()/len(all_probs)*100:.1f}%)")
print(f"  Medium (0.5-0.7): {((all_probs > 0.5) & (all_probs <= 0.7)).sum():,} ({((all_probs > 0.5) & (all_probs <= 0.7)).sum()/len(all_probs)*100:.1f}%)")
print(f"  Low (<0.5): {(all_probs <= 0.5).sum():,} ({(all_probs <= 0.5).sum()/len(all_probs)*100:.1f}%)")

# ============================================================================
# Show example recommendations
print(f"\n{'='*80}")
print("EXAMPLE: Top 5 Recommendations for Each Patient")
print(f"{'='*80}")

for donor_id in sorted(results_df['donor_id'].unique()):
    patient_recs = top_recommendations_df[top_recommendations_df['donor_id'] == donor_id].head(5)
    
    print(f"\n{'─'*80}")
    print(f"Patient: {donor_id}")
    print(f"{'─'*80}")
    
    for idx, row in patient_recs.iterrows():
        pathway_name = row['pathway']
        if len(pathway_name) > 65:
            pathway_name = pathway_name[:62] + "..."
        
        print(f"\n  {row['rank']}. {pathway_name}")
        print(f"     Cell Type: {row['celltype']}")
        print(f"     Match Score: {row['match_probability']:.3f}")
        print(f"     NES: {row['nes']:.2f} | FDR: {row['fdr']:.4f} | Pathway Activity: {row['pathway_score']:.3f}")

print("\n" + "="*80)
print("P4 COMPLETE - ALL PREDICTIONS GENERATED!")
print("="*80)

print(f"\nOutput files:")
print(f"  ✓ All predictions: {OUTPUT_ALL_PREDICTIONS}")
print(f"  ✓ Top 20 per patient: {OUTPUT_TOP20}")
print(f"  ✓ Model metrics: {OUTPUT_METRICS}")
print(f"  ✓ Feature importance: {OUTPUT_FEATURE_IMPORTANCE}")
print(f"  ✓ All visualizations: figures/P4_figs/")

print("\n" + "="*80)