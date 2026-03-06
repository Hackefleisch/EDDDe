import pandas as pd
import itertools
import numpy as np

def main():
    csv_path = '/media/iwe20/DataSSD/QM40_dataset/QM40_main.csv'
    print(f"Loading dataset from {csv_path}...")
    
    # Read the dataset. The first two columns are ID and SMILES, the rest are properties.
    df = pd.read_csv(csv_path)
    
    # Extract property columns (skipping Zinc_id and smile)
    properties = [c for c in df.columns[2:] if c not in ['rot1', 'rot2', 'rot3']]
    df_props = df[properties]
    
    print("Calculating correlation matrix...")
    corr_matrix = df_props.corr()
    
    print("\n--- Correlation Matrix ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(corr_matrix.round(3))
    
    print("\nFinding subset of 4 properties with minimal correlation among themselves...")
    # Finding 4 properties with the lowest max absolute correlation
    best_subset = None
    min_max_corr = float('inf')
    
    for subset in itertools.combinations(properties, 4):
        # Extract the 4x4 submatrix
        sub_corr = corr_matrix.loc[list(subset), list(subset)]
        
        # Get absolute values
        abs_sub_corr = sub_corr.abs()
        
        # We only care about the off-diagonal elements, so set diagonal to 0
        abs_sub_corr_val = abs_sub_corr.values
        np.fill_diagonal(abs_sub_corr_val, 0)
        
        # Find the maximum correlation in this subset
        max_corr_in_subset = abs_sub_corr_val.max()
        
        if max_corr_in_subset < min_max_corr:
            min_max_corr = max_corr_in_subset
            best_subset = subset
            
    print(f"\n--- Best Subset of 4 Properties (min-max absolute correlation: {min_max_corr:.3f}) ---")
    print(list(best_subset))
    print("\nTheir correlation sub-matrix:")
    print(corr_matrix.loc[list(best_subset), list(best_subset)].round(3))

    previous_subset = ['Polarizability', 'dipol_mom', 'CV', 'Entropy']
    print(f"\n--- Previously Used Subset: {previous_subset} ---")
    print("Their correlation sub-matrix:")
    print(corr_matrix.loc[previous_subset, previous_subset].round(3))
    
    # Calculate max off-diagonal abs correlation for comparison
    prev_sub_corr = corr_matrix.loc[previous_subset, previous_subset].abs()
    np.fill_diagonal(prev_sub_corr.values, 0)
    print(f"Max absolute correlation in previously used subset: {prev_sub_corr.values.max():.3f}")

if __name__ == '__main__':
    main()
