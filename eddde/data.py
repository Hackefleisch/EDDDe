"""
Data loading and preprocessing for molecular graph data.

This module provides classes and functions for loading molecular structures
from various formats (SMILES, SDF, XYZ) and converting them to PyTorch Geometric
Data objects suitable for ElektroNN predictions.
"""

from typing import Dict, Optional, Callable, List, Tuple
from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from elektronn.elektronn_ensemble_predict import data_from_rdkit


class MoleculeDataset(Dataset):
    """
    PyTorch Dataset for molecular graph data with lazy 3D embedding.

    This dataset loads molecules from SMILES strings and converts them to
    PyTorch Geometric Data objects on-the-fly. 3D conformer generation is
    performed lazily during __getitem__ to avoid memory overhead.

    Parameters
    ----------
    smiles_dict : Dict[str, str]
        Dictionary mapping molecule names/IDs to SMILES strings
    basisfunction_params : Dict
        Basis function parameters for supported atoms
    conformer_seed : int, optional
        Random seed for conformer generation (default: 42069)
    transform : Callable, optional
        Optional transform to apply to Data objects
    filter_bad_atoms : bool, optional
        Whether to filter molecules with unsupported atoms (default: True)
    verbose : bool, optional
        Whether to print warnings about filtered molecules (default: False)
    cache_conformers : bool, optional
        Whether to cache generated 3D conformers to avoid regeneration (default: False)
    pregenerate_conformers : bool, optional
        Whether to generate all conformers upfront (default: False).
        Only used if cache_conformers is True.

    Attributes
    ----------
    mols : List[Chem.Mol]
        List of RDKit molecule objects (without hydrogens)
    names : List[str]
        List of molecule names/IDs
    bad_atom_count : int
        Number of molecules filtered due to unsupported atoms
    conformer_cache : Dict[int, Chem.Mol]
        Cache of molecules with 3D conformers (if caching enabled)
    """

    def __init__(
        self,
        smiles_dict: Dict[str, str],
        basisfunction_params: Dict,
        conformer_seed: int = 42069,
        transform: Optional[Callable] = None,
        filter_bad_atoms: bool = True,
        verbose: bool = False,
        cache_conformers: bool = False,
        pregenerate_conformers: bool = False,
    ):
        super().__init__()
        self.mols: List[Chem.Mol] = []
        self.names: List[str] = []
        self.basisfunction_params = basisfunction_params
        self.transform = transform
        self.verbose = verbose
        self.cache_conformers = cache_conformers
        self.pregenerate_conformers = pregenerate_conformers

        # Setup conformer generation parameters
        self.conformer_params = AllChem.ETKDGv3()
        self.conformer_params.randomSeed = conformer_seed

        # Track filtered molecules
        self.bad_atom_count = 0

        # Cache for 3D conformers (only used if cache_conformers is True)
        self.conformer_cache: Dict[int, Chem.Mol] = {}

        # Load and validate molecules
        self._load_molecules(smiles_dict, filter_bad_atoms)

        # Optionally pregenerate all conformers
        if self.cache_conformers and self.pregenerate_conformers:
            self._pregenerate_all_conformers()

    def _load_molecules(self, smiles_dict: Dict[str, str], filter_bad_atoms: bool):
        """Load molecules from SMILES and optionally filter unsupported atoms."""
        for name, smiles in smiles_dict.items():
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                if self.verbose:
                    print(
                        f"Warning: Could not parse SMILES for {name}: {smiles}")
                self.bad_atom_count += 1
                continue

            # Check for unsupported atoms
            if filter_bad_atoms:
                has_bad_atom = False
                for atom in mol.GetAtoms():
                    atomic_number = float(atom.GetAtomicNum())
                    if atomic_number not in self.basisfunction_params:
                        if self.verbose:
                            print(
                                f"Warning: Molecule {name} contains unsupported atom (Z={atomic_number})")
                        has_bad_atom = True
                        break

                if has_bad_atom:
                    self.bad_atom_count += 1
                    continue

            self.mols.append(mol)
            self.names.append(name)

    def __len__(self) -> int:
        """Return the number of molecules in the dataset."""
        return len(self.mols)

    def __getitem__(self, index: int) -> Data:
        """
        Get a molecule as a PyTorch Geometric Data object.

        This method performs 3D conformer generation on-the-fly and converts
        the molecule to a graph representation. If caching is enabled, conformers
        are stored after generation to avoid recomputation.

        Parameters
        ----------
        index : int
            Index of the molecule to retrieve

        Returns
        -------
        Data
            PyTorch Geometric Data object with molecular graph
        """
        name = self.names[index]

        # Check if conformer is cached
        if self.cache_conformers and index in self.conformer_cache:
            mol_with_h = self.conformer_cache[index]
        else:
            # Generate new conformer
            mol = self.mols[index]
            mol_with_h = self._generate_conformer(mol, name)

            # Cache if enabled
            if self.cache_conformers:
                self.conformer_cache[index] = mol_with_h

        # Convert to PyTorch Geometric Data
        data_mol = data_from_rdkit(mol_with_h, name, self.basisfunction_params)

        # Apply optional transform
        if self.transform is not None:
            data_mol = self.transform(data_mol)

        return data_mol

    def _generate_conformer(self, mol: Chem.Mol, name: str) -> Chem.Mol:
        """
        Generate 3D conformer for a molecule.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object (without hydrogens)
        name : str
            Name/ID of the molecule (for error messages)

        Returns
        -------
        Chem.Mol
            Molecule with hydrogens and 3D conformer

        Raises
        ------
        RuntimeError
            If conformer generation fails
        """
        # Add hydrogens and generate 3D conformer
        mol_with_h = Chem.AddHs(mol)
        embed_result = AllChem.EmbedMolecule(mol_with_h, self.conformer_params)

        if embed_result == -1:
            # Fallback: try without seed
            if self.verbose:
                print(
                    f"Warning: Failed to embed {name} with seed, trying without seed")
            embed_result = AllChem.EmbedMolecule(mol_with_h)

            if embed_result == -1:
                raise RuntimeError(
                    f"Failed to generate 3D conformer for molecule {name}")

        return mol_with_h

    def _pregenerate_all_conformers(self):
        """
        Generate and cache all conformers upfront.

        This can be useful when you know you'll access all molecules multiple times
        and want to pay the conformer generation cost once at initialization.
        """
        if self.verbose:
            print(
                f"Pre-generating conformers for {len(self.mols)} molecules...")

        from tqdm import tqdm
        iterator = tqdm(enumerate(self.mols), total=len(
            self.mols)) if self.verbose else enumerate(self.mols)

        for idx, mol in iterator:
            name = self.names[idx]
            try:
                mol_with_h = self._generate_conformer(mol, name)
                self.conformer_cache[idx] = mol_with_h
            except RuntimeError as e:
                if self.verbose:
                    print(f"Failed to generate conformer for {name}: {e}")
                # Store None to mark as failed
                self.conformer_cache[idx] = None

        if self.verbose:
            successful = sum(
                1 for v in self.conformer_cache.values() if v is not None)
            print(
                f"Successfully generated {successful}/{len(self.mols)} conformers")

    def get_by_name(self, name: str) -> Data:
        """
        Get a molecule by its name/ID.

        Parameters
        ----------
        name : str
            Name/ID of the molecule

        Returns
        -------
        Data
            PyTorch Geometric Data object

        Raises
        ------
        ValueError
            If molecule name not found in dataset
        """
        try:
            index = self.names.index(name)
            return self[index]
        except ValueError:
            raise ValueError(f"Molecule '{name}' not found in dataset")

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns
        -------
        Dict
            Dictionary with dataset statistics including number of molecules,
            filtered molecules, and atom counts
        """
        total_atoms = sum(mol.GetNumAtoms() for mol in self.mols)
        stats = {
            "num_molecules": len(self.mols),
            "num_filtered": self.bad_atom_count,
            "total_atoms": total_atoms,
            "avg_atoms_per_mol": total_atoms / len(self.mols) if self.mols else 0,
            "cache_enabled": self.cache_conformers,
            "cached_conformers": len(self.conformer_cache),
        }
        return stats

    def clear_cache(self):
        """
        Clear the conformer cache to free memory.

        This can be useful after processing a dataset once if you don't
        need to access it again.
        """
        self.conformer_cache.clear()
        if self.verbose:
            print("Conformer cache cleared")

    def get_cached_conformer(self, index: int) -> Optional[Chem.Mol]:
        """
        Get a cached conformer by index without generating Data object.

        Parameters
        ----------
        index : int
            Index of the molecule

        Returns
        -------
        Chem.Mol or None
            Cached molecule with conformer, or None if not cached
        """
        return self.conformer_cache.get(index, None)


def load_smiles_csv(
    filepath: Path,
    smiles_col: str = "SMILES",
    id_col: str = "CID",
    delimiter: str = ",",
) -> Dict[str, str]:
    """
    Load SMILES strings from a CSV file.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file
    smiles_col : str, optional
        Name of the column containing SMILES strings (default: 'SMILES')
    id_col : str, optional
        Name of the column containing molecule IDs (default: 'CID')
    delimiter : str, optional
        CSV delimiter (default: ',')

    Returns
    -------
    Dict[str, str]
        Dictionary mapping molecule IDs to SMILES strings

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    KeyError
        If required columns are not found in the CSV
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    smiles_dict = {}

    with open(filepath, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)

        # Validate columns
        if smiles_col not in reader.fieldnames:
            raise KeyError(
                f"Column '{smiles_col}' not found in CSV. Available columns: {reader.fieldnames}")
        if id_col not in reader.fieldnames:
            raise KeyError(
                f"Column '{id_col}' not found in CSV. Available columns: {reader.fieldnames}")

        for row in reader:
            mol_id = row[id_col]
            smiles = row[smiles_col]
            smiles_dict[mol_id] = smiles

    return smiles_dict


def load_molecules_from_sdf(
    sdf_path: Path,
    remove_hydrogens: bool = False,
) -> Dict[str, Chem.Mol]:
    """
    Load molecules from an SDF file.

    SDF files already contain 3D conformers, so no conformer generation is needed.

    Parameters
    ----------
    sdf_path : Path
        Path to the SDF file
    remove_hydrogens : bool, optional
        Whether to remove hydrogens from molecules (default: False)

    Returns
    -------
    Dict[str, Chem.Mol]
        Dictionary mapping molecule names to RDKit Mol objects with conformers

    Raises
    ------
    FileNotFoundError
        If the SDF file does not exist
    """
    sdf_path = Path(sdf_path)
    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    molecules = {}

    # Load molecules from SDF
    suppl = Chem.SDMolSupplier(
        str(sdf_path), sanitize=False, removeHs=remove_hydrogens)

    for idx, mol in enumerate(suppl):
        if mol is None:
            continue

        # Try to get molecule name from properties
        mol_name = None
        if mol.HasProp("_Name"):
            mol_name = mol.GetProp("_Name")
        elif mol.HasProp("PUBCHEM_COMPOUND_CID"):
            mol_name = mol.GetProp("PUBCHEM_COMPOUND_CID")
        elif mol.HasProp("ID"):
            mol_name = mol.GetProp("ID")

        # Fallback to index
        if not mol_name:
            mol_name = str(idx)

        molecules[mol_name] = mol

    return molecules


class SDFMoleculeDataset(Dataset):
    """
    PyTorch Dataset for molecules loaded from SDF files.

    Unlike MoleculeDataset, this dataset loads molecules that already have
    3D conformers from the SDF file, so no conformer generation is needed.

    Parameters
    ----------
    sdf_path : Path
        Path to SDF file
    basisfunction_params : Dict
        Basis function parameters for supported atoms
    transform : Callable, optional
        Optional transform to apply to Data objects
    filter_bad_atoms : bool, optional
        Whether to filter molecules with unsupported atoms (default: True)
    verbose : bool, optional
        Whether to print warnings (default: False)
    name_mapping : Dict[str, str], optional
        Optional mapping from SDF molecule names to desired names

    Attributes
    ----------
    mols : Dict[str, Chem.Mol]
        Dictionary mapping names to RDKit Mol objects with conformers
    names : List[str]
        List of molecule names
    bad_atom_count : int
        Number of molecules filtered due to unsupported atoms
    """

    def __init__(
        self,
        sdf_path: Path,
        basisfunction_params: Dict,
        transform: Optional[Callable] = None,
        filter_bad_atoms: bool = True,
        verbose: bool = False,
        name_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.basisfunction_params = basisfunction_params
        self.transform = transform
        self.verbose = verbose
        self.name_mapping = name_mapping or {}

        # Load molecules from SDF
        molecules = load_molecules_from_sdf(sdf_path)

        # Filter and store
        self.mols: Dict[str, Chem.Mol] = {}
        self.names: List[str] = []
        self.bad_atom_count = 0

        for name, mol in molecules.items():
            # Apply name mapping if provided
            mapped_name = self.name_mapping.get(name, name)

            # Check for unsupported atoms
            if filter_bad_atoms:
                has_bad_atom = False
                for atom in mol.GetAtoms():
                    atomic_number = float(atom.GetAtomicNum())
                    if atomic_number not in self.basisfunction_params:
                        if self.verbose:
                            print(
                                f"Warning: Molecule {mapped_name} contains unsupported atom (Z={atomic_number})")
                        has_bad_atom = True
                        break

                if has_bad_atom:
                    self.bad_atom_count += 1
                    continue

            self.mols[mapped_name] = mol
            self.names.append(mapped_name)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Data:
        """
        Get a molecule as a PyTorch Geometric Data object.

        Parameters
        ----------
        index : int
            Index of the molecule to retrieve

        Returns
        -------
        Data
            PyTorch Geometric Data object with molecular graph
        """
        name = self.names[index]
        mol = self.mols[name]

        # Convert to PyTorch Geometric Data
        # Molecule already has conformer from SDF
        data_mol = data_from_rdkit(mol, name, self.basisfunction_params)

        # Apply optional transform
        if self.transform is not None:
            data_mol = self.transform(data_mol)

        return data_mol

    def get_by_name(self, name: str) -> Data:
        """Get a molecule by its name."""
        try:
            index = self.names.index(name)
            return self[index]
        except ValueError:
            raise ValueError(f"Molecule '{name}' not found in dataset")

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        total_atoms = sum(mol.GetNumAtoms() for mol in self.mols.values())
        return {
            "num_molecules": len(self.names),
            "num_filtered": self.bad_atom_count,
            "total_atoms": total_atoms,
            "avg_atoms_per_mol": total_atoms / len(self.names) if self.names else 0,
        }


def create_dataset_from_csv(
    csv_path: Path,
    basisfunction_params: Dict,
    smiles_col: str = "SMILES",
    id_col: str = "CID",
    **dataset_kwargs,
) -> MoleculeDataset:
    """
    Convenience function to create a MoleculeDataset from a CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file containing SMILES
    basisfunction_params : Dict
        Basis function parameters
    smiles_col : str, optional
        SMILES column name (default: 'SMILES')
    id_col : str, optional
        ID column name (default: 'CID')
    **dataset_kwargs
        Additional keyword arguments passed to MoleculeDataset

    Returns
    -------
    MoleculeDataset
        Dataset object ready for use
    """
    smiles_dict = load_smiles_csv(csv_path, smiles_col, id_col)
    return MoleculeDataset(smiles_dict, basisfunction_params, **dataset_kwargs)


def create_dataset_from_sdf(
    sdf_path: Path,
    basisfunction_params: Dict,
    **dataset_kwargs,
) -> SDFMoleculeDataset:
    """
    Convenience function to create an SDFMoleculeDataset from an SDF file.

    Parameters
    ----------
    sdf_path : Path
        Path to SDF file
    basisfunction_params : Dict
        Basis function parameters
    **dataset_kwargs
        Additional keyword arguments passed to SDFMoleculeDataset

    Returns
    -------
    SDFMoleculeDataset
        Dataset object ready for use
    """
    return SDFMoleculeDataset(sdf_path, basisfunction_params, **dataset_kwargs)


def convert_smiles_to_sdf(
    smiles_dict: Dict[str, str],
    output_sdf_path: Path,
    conformer_seed: int = 42069,
    add_hydrogens: bool = True,
    optimize_geometry: bool = False,
    verbose: bool = True,
    filter_bad_atoms: bool = False,
    basisfunction_params: Optional[Dict] = None,
) -> Tuple[int, int]:
    """
    Convert SMILES strings to 3D conformers and write to SDF file.

    This function generates 3D conformers for molecules from SMILES strings
    and saves them to an SDF file. This is useful for pre-computing conformers
    once and reusing them later for faster inference.

    Parameters
    ----------
    smiles_dict : Dict[str, str]
        Dictionary mapping molecule IDs to SMILES strings
    output_sdf_path : Path
        Path where the SDF file will be written
    conformer_seed : int, optional
        Random seed for conformer generation (default: 42069)
    add_hydrogens : bool, optional
        Whether to add explicit hydrogens (default: True)
    optimize_geometry : bool, optional
        Whether to optimize geometry with MMFF (default: False)
    verbose : bool, optional
        Whether to print progress information (default: True)
    filter_bad_atoms : bool, optional
        Whether to skip molecules with unsupported atoms (default: False)
        Requires basisfunction_params to be provided
    basisfunction_params : Dict, optional
        Basis function parameters for supported atoms (required if filter_bad_atoms=True)

    Returns
    -------
    Tuple[int, int]
        Number of successfully converted molecules and number of failed molecules

    Raises
    ------
    ValueError
        If filter_bad_atoms is True but basisfunction_params is not provided

    Examples
    --------
    >>> smiles = {"mol1": "CCO", "mol2": "c1ccccc1"}
    >>> convert_smiles_to_sdf(smiles, "output.sdf")
    (2, 0)
    """
    if filter_bad_atoms and basisfunction_params is None:
        raise ValueError(
            "basisfunction_params must be provided when filter_bad_atoms=True"
        )

    output_sdf_path = Path(output_sdf_path)
    output_sdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup conformer generation parameters
    conformer_params = AllChem.ETKDGv3()
    conformer_params.randomSeed = conformer_seed

    success_count = 0
    failed_count = 0

    # Open SDF writer
    writer = Chem.SDWriter(str(output_sdf_path))

    # Setup progress bar if verbose
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(smiles_dict.items(),
                            desc="Converting SMILES to SDF")
        except ImportError:
            iterator = smiles_dict.items()
            print(f"Converting {len(smiles_dict)} molecules to SDF...")
    else:
        iterator = smiles_dict.items()

    for mol_id, smiles in iterator:
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                if verbose:
                    print(
                        f"Warning: Could not parse SMILES for {mol_id}: {smiles}")
                failed_count += 1
                continue

            # Check for unsupported atoms
            if filter_bad_atoms:
                has_bad_atom = False
                for atom in mol.GetAtoms():
                    atomic_number = float(atom.GetAtomicNum())
                    if atomic_number not in basisfunction_params:
                        if verbose:
                            print(
                                f"Warning: Molecule {mol_id} contains unsupported atom (Z={atomic_number})"
                            )
                        has_bad_atom = True
                        break

                if has_bad_atom:
                    failed_count += 1
                    continue

            # Add hydrogens if requested
            if add_hydrogens:
                mol = Chem.AddHs(mol)

            # Generate 3D conformer
            embed_result = AllChem.EmbedMolecule(mol, conformer_params)

            if embed_result == -1:
                # Fallback: try without seed
                if verbose:
                    print(
                        f"Warning: Failed to embed {mol_id} with seed, trying without seed"
                    )
                embed_result = AllChem.EmbedMolecule(mol)

                if embed_result == -1:
                    if verbose:
                        print(
                            f"Warning: Failed to generate 3D conformer for {mol_id}")
                    failed_count += 1
                    continue

            # Optimize geometry if requested
            if optimize_geometry:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception as e:
                    if verbose:
                        print(
                            f"Warning: Geometry optimization failed for {mol_id}: {e}")

            # Set molecule name
            mol.SetProp("_Name", str(mol_id))

            # Write to SDF
            writer.write(mol)
            success_count += 1

        except Exception as e:
            if verbose:
                print(f"Error processing molecule {mol_id}: {e}")
            failed_count += 1

    # Close writer
    writer.close()

    if verbose:
        print(f"\nConversion complete:")
        print(f"  Successfully converted: {success_count} molecules")
        print(f"  Failed: {failed_count} molecules")
        print(f"  Output file: {output_sdf_path}")

    return success_count, failed_count


def convert_csv_to_sdf(
    csv_path: Path,
    output_sdf_path: Path,
    smiles_col: str = "SMILES",
    id_col: str = "CID",
    delimiter: str = ",",
    **converter_kwargs,
) -> Tuple[int, int]:
    """
    Convert molecules from CSV with SMILES to SDF file with 3D conformers.

    This is a convenience wrapper around convert_smiles_to_sdf that loads
    SMILES from a CSV file first.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file containing SMILES
    output_sdf_path : Path
        Path where the SDF file will be written
    smiles_col : str, optional
        Name of the column containing SMILES strings (default: 'SMILES')
    id_col : str, optional
        Name of the column containing molecule IDs (default: 'CID')
    delimiter : str, optional
        CSV delimiter (default: ',')
    **converter_kwargs
        Additional keyword arguments passed to convert_smiles_to_sdf

    Returns
    -------
    Tuple[int, int]
        Number of successfully converted molecules and number of failed molecules

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    KeyError
        If required columns are not found in the CSV

    Examples
    --------
    >>> convert_csv_to_sdf(
    ...     "molecules.csv",
    ...     "molecules.sdf",
    ...     smiles_col="SMILES",
    ...     id_col="ID",
    ... )
    (100, 5)
    """
    # Load SMILES from CSV
    smiles_dict = load_smiles_csv(csv_path, smiles_col, id_col, delimiter)

    # Convert to SDF
    return convert_smiles_to_sdf(smiles_dict, output_sdf_path, **converter_kwargs)
