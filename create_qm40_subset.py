import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
import argparse
import os
from scipy.stats import pearsonr, spearmanr

def get_heavy_atom_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return mol.GetNumHeavyAtoms()
    return None

def create_subset(target_col, subset_size=10000, input_csv='/media/iwe20/DataSSD/QM40_dataset/QM40_main.csv'):
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Calculating heavy atom counts for {len(df)} molecules...")
    df['heavy_atom_count'] = df['smile'].apply(get_heavy_atom_count)
    df = df.dropna(subset=['heavy_atom_count', target_col])
    
    # Stratified sampling based on 2D binning of target and heavy_atom_count
    print(f"Performing stratified sampling for target: {target_col}...")
    
    # Define bins for stratification
    n_bins = 20
    df['target_bin'] = pd.qcut(df[target_col], n_bins, labels=False, duplicates='drop')
    df['ha_bin'] = pd.cut(df['heavy_atom_count'], bins=range(0, int(df['heavy_atom_count'].max()) + 2, 2), labels=False)
    
    # Combine bins for 2D stratification
    df['strata'] = df['target_bin'].astype(str) + "_" + df['ha_bin'].astype(str)
    
    # Calculate weights for sampling based on strata frequency in original set
    strata_counts = df['strata'].value_counts()
    
    # We want to maintain the same distribution, so we sample proportionally to strata frequency
    subset_indices = []
    for f_name, group in df.groupby('strata'):
        n_samples = int(np.round(len(group) * subset_size / len(df)))
        if n_samples > 0:
            subset_indices.extend(group.sample(n=min(n_samples, len(group)), replace=False).index)
    
    subset = df.loc[subset_indices]
    
    # Final adjustment to match exact subset_size
    if len(subset) > subset_size:
        subset = subset.sample(n=subset_size, replace=False)
    elif len(subset) < subset_size:
        # Fill the gap with random samples from the remaining data
        remaining = df.drop(subset_indices)
        gap = subset_size - len(subset)
        if gap > 0 and len(remaining) > 0:
            extra = remaining.sample(n=min(gap, len(remaining)), replace=False)
            subset = pd.concat([subset, extra])
    
    # Reporting
    print(f"\n--- Statistics for {target_col} ---")
    stats = pd.DataFrame({
        'Original': df[target_col].describe(),
        'Subset': subset[target_col].describe()
    })
    print(stats)
    
    orig_corr, _ = pearsonr(df['heavy_atom_count'], df[target_col])
    sub_corr, _ = pearsonr(subset['heavy_atom_count'], subset[target_col])
    print(f"\nPearson correlation (Heavy Atoms vs {target_col}):")
    print(f"Original: {orig_corr:.4f}")
    print(f"Subset:   {sub_corr:.4f}")
    
    # Plotting
    print(f"Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot with density kernel for Original
    sns.kdeplot(data=df, x='heavy_atom_count', y=target_col, ax=axes[0, 0], cmap="Blues", fill=True, thresh=0.05)
    axes[0, 0].set_title(f"Original: Heavy Atoms vs {target_col} (Density)")
    
    # Scatter plot with density kernel for Subset
    sns.kdeplot(data=subset, x='heavy_atom_count', y=target_col, ax=axes[0, 1], cmap="Oranges", fill=True, thresh=0.05)
    axes[0, 1].set_title(f"Subset: Heavy Atoms vs {target_col} (Density)")
    
    # Violin plots for distribution comparison
    plot_df = pd.concat([
        df[[target_col]].assign(Dataset='Original'),
        subset[[target_col]].assign(Dataset='Subset')
    ])
    sns.violinplot(data=plot_df, x='Dataset', y=target_col, ax=axes[1, 0])
    axes[1, 0].set_title(f"Distribution Comparison: {target_col}")
    
    # Correlation scatter (Subset only for clarity)
    sns.regplot(data=subset, x='heavy_atom_count', y=target_col, scatter_kws={'alpha':0.1, 's':1}, line_kws={'color':'red'}, ax=axes[1, 1])
    axes[1, 1].set_title(f"Subset: Correlation Trendline")
    
    plt.tight_layout()
    plot_name = f"{target_col}_comparison_plot.png"
    plt.savefig(plot_name)
    print(f"Plot saved as {plot_name}")
    
    # Save subset
    output_name = f"QM40_subset_{target_col}.csv"
    subset.drop(columns=['target_bin', 'ha_bin', 'strata']).to_csv(output_name, index=False)
    print(f"Subset saved as {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a subset of QM40 dataset.')
    parser.add_argument('--target', type=str, required=True, help='Target property (e.g., HOMO, LUMO, dipol_mom, Entropy)')
    parser.add_argument('--size', type=int, default=10000, help='Size of the subset')
    
    args = parser.parse_args()
    create_subset(args.target, args.size)
