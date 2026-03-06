import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import argparse
import sys

# Initialize the default RDKit fingerprint generator
# This avoids deprecation warnings and uses default path-based settings
fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator()

def smiles_to_fp(smiles):
    """Convert SMILES to the default RDKit topological fingerprint numpy array."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Use the modern generator API
        fp = fp_gen.GetFingerprint(mol)
        arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert SMILES to RDKit fingerprints and save to .npy")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of processed entries (for testing)")
    parser.add_argument("--csv", type=str, default='/media/iwe20/DataSSD/QM40_dataset/QM40_main.csv', help="Path to input CSV")
    parser.add_argument("--output", type=str, default='qm40_fingerprints.npy', help="Path to output .npy file")
    args = parser.parse_args()

    print(f"Loading data from {args.csv}...")
    try:
        # Load only necessary columns
        df = pd.read_csv(args.csv, usecols=['Zinc_id', 'smile'], nrows=args.limit)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        sys.exit(1)
    
    fp_dict = {}
    print(f"Generating RDKit topological fingerprints (default settings) for {len(df)} entries...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        zinc_id = row['Zinc_id']
        smiles = row['smile']
        fp = smiles_to_fp(smiles)
        if fp is not None:
            fp_dict[zinc_id] = fp

    print(f"Saving {len(fp_dict)} fingerprints to {args.output}...")
    # Saving as a dictionary inside the .npy file
    np.save(args.output, fp_dict)
    
    print("\nVerification: Loading back the file...")
    loaded_dict = np.load(args.output, allow_pickle=True).item()
    test_id = df.iloc[0]['Zinc_id']
    if test_id in loaded_dict:
        print(f"Success! Retrieved fingerprint for {test_id}. Shape: {loaded_dict[test_id].shape}")
    else:
        print(f"Failed to retrieve {test_id} from saved file.")

if __name__ == "__main__":
    main()
