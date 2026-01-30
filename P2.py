# P2.py - Population-Level Drug2Cell Analysis for SMALL INTESTINE ONLY

import scanpy as sc
import blitzgsea as blitz
import drug2cell as d2c
import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

def setup_directories():
    """Create organized directory structure"""
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("figures/P2_figs", exist_ok=True)
    os.makedirs("figures/P3_figs", exist_ok=True)
    os.makedirs("figures/P4_figs", exist_ok=True)
    os.makedirs("results/P3_results", exist_ok=True)
    os.makedirs("results/P4_results", exist_ok=True)

def main():
    print("="*80)
    print("P2: POPULATION-LEVEL DRUG2CELL ANALYSIS - SMALL INTESTINE")
    print("="*80)

    MIN_NES = 1.5
    MAX_FDR = 0.25

    print(f"\nFiltering: |NES| > {MIN_NES}, FDR < {MAX_FDR}")

    setup_directories()

    # Small intestine checkpoints
    checkpoint_filtered = "checkpoints/P2_filtered_data.h5ad"
    checkpoint_drug2cell = "checkpoints/P2_drug2cell_scored.h5ad"
    checkpoint_ranked = "checkpoints/P2_ranked.h5ad"
    checkpoint_gsea = "checkpoints/P2_gsea_results.csv"
    checkpoint_gsea_filtered = "checkpoints/P2_gsea_results_BIG_DARK_ONLY.csv"
    checkpoint_hyper = "checkpoints/P2_hypergeometric_results.csv"
    checkpoint_hyper_filtered = "checkpoints/P2_hypergeometric_results_BIG_DARK_ONLY.csv"

    # Check if GSEA needs regeneration
    needs_regeneration = False
    if os.path.exists(checkpoint_gsea_filtered):
        test_df = pd.read_csv(checkpoint_gsea_filtered)
        if 'pathway' not in test_df.columns:
            print("\n⚠️  GSEA missing 'pathway' column - regenerating...")
            needs_regeneration = True
            os.remove(checkpoint_gsea_filtered)
            if os.path.exists(checkpoint_gsea):
                os.remove(checkpoint_gsea)

    # STEP 1: Load and filter data
    if os.path.exists(checkpoint_filtered):
        print("\n[1/7] Loading filtered data...")
        adata = sc.read_h5ad(checkpoint_filtered)
        print(f"      Loaded: {adata.shape}")
    else:
        print("\n[1/7] Loading and filtering small intestine data...")
        adata = sc.read_h5ad("/Users/rvellamcheti/Downloads/Cleaned_raw_annotated_object_LK.v2.h5ad")
        
        # Filter for Crohn's disease
        adata_crohns = adata[adata.obs['disease'].isin(["MONDO_0005011"]), :].copy()
        
        # Filter by SMALL INTESTINE tissue
        adata_tissue = adata_crohns[adata_crohns.obs['tissue'].isin(["small_intestine"]), :].copy()
        
        # Keep cell types with ≥100 cells
        celltype_counts = adata_tissue.obs['annotation2v2'].value_counts()
        valid_celltypes = celltype_counts[celltype_counts >= 100].index.tolist()
        adata_tissue = adata_tissue[adata_tissue.obs['annotation2v2'].isin(valid_celltypes), :].copy()
        
        adata = adata_tissue
        adata.write(checkpoint_filtered)
        print(f"      Saved: {adata.shape}")

    # STEP 2: UMAP
    if not os.path.exists("figures/P2_figs/umap_P2_cell_types.png"):
        print("\n[2/7] Generating UMAP...")
        sc.settings.set_figure_params(dpi=80)
        sc.pl.umap(adata, color="annotation2v2", save="_P2_cell_types.png", show=False)
    else:
        print("\n[2/7] UMAP exists, skipping...")

    # STEP 3: Drug2cell scoring
    if os.path.exists(checkpoint_drug2cell):
        print("\n[3/7] Loading drug2cell scored data...")
        adata = sc.read_h5ad(checkpoint_drug2cell)
    else:
        print("\n[3/7] Running drug2cell scoring...")
        d2c.score(adata, use_raw=True)
        adata.write(checkpoint_drug2cell)

    # STEP 4: Rank genes
    if os.path.exists(checkpoint_ranked):
        print("\n[4/7] Loading ranked data...")
        adata = sc.read_h5ad(checkpoint_ranked)
    else:
        print("\n[4/7] Ranking genes...")
        sc.tl.rank_genes_groups(adata.uns['drug2cell'], method="wilcoxon", groupby="annotation2v2")
        
        adata_log = adata.copy()
        sc.pp.normalize_total(adata_log, target_sum=1e4)
        sc.pp.log1p(adata_log)
        sc.tl.rank_genes_groups(adata_log, method="wilcoxon", groupby="annotation2v2")
        adata.uns['rank_genes_groups'] = adata_log.uns['rank_genes_groups']
        del adata_log
        
        adata.write(checkpoint_ranked)

    # STEP 5: Hypergeometric
    if os.path.exists(checkpoint_hyper_filtered):
        print("\n[5/7] Hypergeometric results exist, skipping...")
    else:
        print("\n[5/7] Running hypergeometric test...")
        
        if not os.path.exists(checkpoint_hyper):
            targets = blitz.enrichr.get_library("GO_Molecular_Function_2021")
            d2c.score(adata, targets=targets, use_raw=True)
            overrepresentation = d2c.hypergeometric(adata)
            
            for celltype, results in overrepresentation.items():
                results['celltype'] = celltype
            
            all_hyper = pd.concat(overrepresentation.values(), ignore_index=True)
            all_hyper.to_csv(checkpoint_hyper, index=False)
        else:
            all_hyper = pd.read_csv(checkpoint_hyper)
        
        filtered_hyper = all_hyper[all_hyper['pvals'] < 0.05].copy()
        filtered_hyper.to_csv(checkpoint_hyper_filtered, index=False)
        print(f"      Saved {len(filtered_hyper)} filtered results")

    # STEP 6: GSEA with pathway names
    if os.path.exists(checkpoint_gsea_filtered) and not needs_regeneration:
        print("\n[6/7] GSEA results exist, skipping...")
    else:
        print("\n[6/7] Running GSEA with pathway extraction...")
        
        targets = blitz.enrichr.get_library("GO_Molecular_Function_2021")
        
        if 'rank_genes_groups' not in adata.uns:
            print("      Re-ranking genes...")
            adata_log = adata.copy()
            sc.pp.normalize_total(adata_log, target_sum=1e4)
            sc.pp.log1p(adata_log)
            sc.tl.rank_genes_groups(adata_log, method="wilcoxon", groupby="annotation2v2")
            adata.uns['rank_genes_groups'] = adata_log.uns['rank_genes_groups']
            del adata_log
        
        print("      Running GSEA (30-60 minutes)...")
        enrichment, plot_gsea_args = d2c.gsea(adata, targets=targets)
        print("      GSEA complete!")
        
        # CRITICAL: Extract pathway names from DataFrame index
        print("      Extracting pathway names...")
        all_gsea_list = []
        
        for celltype, results_df in enrichment.items():
            results_df = results_df.copy()
            results_df['pathway'] = results_df.index  # Extract from index!
            results_df['celltype'] = celltype
            results_df = results_df.reset_index(drop=True)
            all_gsea_list.append(results_df)
        
        all_gsea = pd.concat(all_gsea_list, ignore_index=True)
        
        # Verify
        if 'pathway' in all_gsea.columns:
            print(f"      ✓ 'pathway' column created ({all_gsea['pathway'].nunique()} unique)")
        else:
            raise ValueError("ERROR: 'pathway' column missing!")
        
        all_gsea.to_csv(checkpoint_gsea, index=False)
        print(f"      Saved {len(all_gsea)} total results")
        
        # Filter
        filtered_gsea = all_gsea[
            (all_gsea['fdr'] < MAX_FDR) &
            (abs(all_gsea['nes']) > MIN_NES)
        ].copy()
        
        print(f"      Filtered: {len(filtered_gsea)} enrichments")
        filtered_gsea.to_csv(checkpoint_gsea_filtered, index=False)
        
        if 'pathway' not in filtered_gsea.columns:
            raise ValueError("ERROR: Pathway column lost during filtering!")
        print("Fixing P2 GSEA infinite NES values...")

        # Load P2 data
        gsea = pd.read_csv("checkpoints/P2_gsea_results_BIG_DARK_ONLY.csv")

        print(f"Before: {len(gsea)} rows")
        print(f"Infinite NES values: {np.isinf(gsea['nes']).sum()}")

        # Remove rows with infinite NES
        gsea_clean = gsea[~np.isinf(gsea['nes'])].copy()

        print(f"After: {len(gsea_clean)} rows")
        print(f"New NES range: [{gsea_clean['nes'].min():.2f}, {gsea_clean['nes'].max():.2f}]")

        # Save cleaned version
        gsea_clean.to_csv("checkpoints/P2_gsea_results_BIG_DARK_ONLY.csv", index=False)
        print("Saved cleaned P2 data")
        
        

    # STEP 7: Summary
    print("\n[7/7] Summary...")
    
    print("\n" + "="*80)
    print("P2 COMPLETE - SMALL INTESTINE")
    print("="*80)
    
    print(f"\nDataset: {adata.shape[0]} cells, {adata.obs['annotation2v2'].nunique()} cell types")
    print(f"Donors: {adata.obs['donor_id'].nunique()}")
    
    if os.path.exists(checkpoint_gsea_filtered):
        df = pd.read_csv(checkpoint_gsea_filtered)
        print(f"\nGSEA: {len(df)} filtered enrichments")
        
        if 'pathway' in df.columns:
            print(f"✓ Pathway column present ({df['pathway'].nunique()} unique pathways)")
            print("\nTop 3 enrichments:")
            for idx, row in df.nlargest(3, 'nes').iterrows():
                pathway = row['pathway'][:50] + "..." if len(row['pathway']) > 50 else row['pathway']
                print(f"  {row['celltype']}: {pathway} (NES={row['nes']:.2f})")
        else:
            print("✗ ERROR: Pathway column MISSING!")
    
    if os.path.exists(checkpoint_hyper_filtered):
        df_hyper = pd.read_csv(checkpoint_hyper_filtered)
        print(f"\nHypergeometric: {len(df_hyper)} filtered enrichments")
    
    print("\n" + "="*80)
    print("Ready for P3!")
    print("="*80)

if __name__ == '__main__':
    main()