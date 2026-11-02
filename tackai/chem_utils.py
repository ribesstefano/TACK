from typing import List, Optional, Dict

from rdkit import Chem, RDLogger

from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(
        smiles: str,
        return_mol: bool = False,
        unique_inchikeys: Optional[Dict[str, Chem.Mol]] = None,
) -> Optional[str]:
    """ Convert a SMILES string to its canonical form using RDKit.
    
    Args:
        smiles (str): The input SMILES string to canonicalize.
        return_mol (bool, optional): If True, return the RDKit Mol object instead of the SMILES string.
        unique_inchikeys (dict, optional): A dictionary to track unique InChI.
            If provided, the function will return None for any molecule whose
            InChIKey is already in the dictionary, effectively filtering out
            duplicates. The dictionary is updated with new InChIKeys for unique
            molecules.
            
    Returns:
        str or None: The canonical SMILES string, or None if invalid or duplicate.
    """
    if not isinstance(smiles, str):
        # Covers None and the float NaN a CSV read leaves in an empty cell.
        return None

    text = smiles.strip()
    if not text:
        # Must be checked *after* stripping: Chem.MolFromSmiles('') returns an
        # empty Mol rather than None, so a whitespace-only cell would otherwise
        # canonicalize to '' -- falsy, not None, so it slips past an
        # `if result is None` check and lands in a CSV as a blank SMILES.
        return None

    mol = Chem.MolFromSmiles(text, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    try:
        # Keep only the largest fragment (i.e., remove salts)
        mol = rdMolStandardize.FragmentParent(mol)
        
        # Uncharge the molecule (i.e., remove formal charges)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        if unique_inchikeys is not None:
            inchikey = Chem.MolToInchiKey(mol)
            if inchikey in unique_inchikeys:
                return None
            unique_inchikeys[inchikey] = mol

        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if return_mol:
            return Chem.MolFromSmiles(smi)
        return smi
    except Exception:
        return None