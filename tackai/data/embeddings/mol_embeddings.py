""" Handles molecular embeddings using RDKit and Hugging Face Transformers. """
import logging
from pathlib import Path
from typing import Optional, List, Union, Literal, Dict, Any

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from tackai.config import config
from tackai.data.embeddings.utils import EmbeddingMixin

class MolEmbedding(EmbeddingMixin):
    """ Class for handling molecular embeddings. """
    
    def __init__(
        self,
        embeddings_type: Literal["fingerprint", "rdkit_descriptors", "transformer"] = "fingerprint",
        # Fingerprint-specific configurations
        radius: int = config.morgan_radius,
        fp_size: int = config.fingerprint_size,
        use_relevant_descriptors: bool = False,
        selected_descriptors: Optional[List[str]] = None,
        # Transformer configurations
        pretrained_model: str = "ailab-bio/PROTAC-Splitter-Encoder",
        batch_size: int = 64,
        device: Union[int, str] = "cpu",
        pooling: Literal["cls", "mean", "sum", "max", "mean_sqrt_len"] = "sum",
        return_tensors: Literal["pt", "np"] = "np",
        # EmbeddingMixin parameters
        embeddings: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None,
        model: Optional[Union[AutoModel, str]] = None,
        tokenizer: Optional[Union[AutoTokenizer, str]] = None,
        load_from_cache: bool = False,
        filename: Optional[Union[Path, str]] = None,
        cache_dir: Optional[Union[Path, str]] = None,
    ):
        """ Initialize the MolEmbedding class.
        
        Args:
            embeddings_type: Type of embeddings to compute consistently for this instance
            radius: Radius for Morgan fingerprints (only used if embeddings_type="fingerprint")
            fp_size: Size of the Morgan fingerprints (only used if embeddings_type="fingerprint")
            morgan_fpgen: Predefined Morgan fingerprint generator (only used if embeddings_type="fingerprint")
            pretrained_model: Name of the pre-trained model to use if tokenizer or model is None
            batch_size: Batch size for transformer encoding
            device: Device to run the model on ("cpu" or "cuda")
            pooling: Pooling method for transformer embeddings
            return_tensors: Format of the returned tensors
            embeddings: Precomputed embeddings or fingerprints
            model: Pre-trained transformer model for embeddings
            tokenizer: Tokenizer for the transformer model
            load_from_cache: Whether to load embeddings from cache
            filename: Path to the file containing embeddings
            cache_dir: Directory to store cached embeddings
        """
        # Set default filename based on embeddings_type if not provided
        if filename is None:
            filename = f"mol_embeddings_{embeddings_type}.npz"
        else:
            # Check that the filename ends with ".npz"
            if filename.split(".")[-1] != "npz":
                raise ValueError(f"Provided embedding filename must end with '.npz'. Provided: '{filename}'")
        
        super().__init__(
            embeddings=embeddings,
            model=model,
            tokenizer=tokenizer,
            load_from_cache=load_from_cache,
            filename=filename,
            cache_dir=cache_dir,
        )
        
        # Store embedding configuration
        self.embeddings_type = embeddings_type
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.device = device
        self.pooling = pooling
        self.return_tensors = return_tensors
        
        # Initialize fingerprint-specific components
        self.radius = radius
        self.fp_size = fp_size
        self.use_relevant_descriptors = use_relevant_descriptors
        self.selected_descriptors = selected_descriptors
        
        # Cache the Morgan fingerprint generator to avoid recreating it on every call
        self._morgan_fpgen = None
        if embeddings_type == "fingerprint":
            self._morgan_fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
                radius=radius,
                fpSize=fp_size,
                includeChirality=True,
            )
    
    def get_descriptor_names(self) -> List[str]:
        """ Get the list of RDKit descriptor names used in this embedding. """
        if self.embeddings_type != "rdkit_descriptors":
            raise ValueError("Descriptor names are only available for embeddings_type='rdkit_descriptors'.")
        if self.use_relevant_descriptors:
            return RELEVANT_RDKIT_DESCRIPTORS
        elif self.selected_descriptors is not None:
            return self.selected_descriptors
        else:
            return [name for name, _ in Descriptors._descList]

    def transform(
        self,
        smiles: Union[str, List[str]],
        skip_existing: bool = True,
        update_cache: bool = False,
    ) -> Dict[str, Union[np.array, torch.Tensor]]:
        """ Encode SMILES strings into fingerprints or embeddings using the configured method.
        
        Args:
            smiles: SMILES string or list of SMILES strings
            skip_existing: Whether to skip existing embeddings in the cache
            update_cache: Whether to update the cache with the new embeddings
            
        Returns:
            Dict[str, Union[np.array, torch.Tensor]]: Encoded embeddings
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
        elif isinstance(smiles, list):
            smiles_list = smiles
        else:
            raise ValueError("Input smiles must be a string or a list of strings.")

        if skip_existing:
            smiles_to_encode = [s for s in smiles_list if s not in self.embeddings]
            smiles_encoded = {s: self.embeddings[s] for s in smiles_list if s in self.embeddings}
        else:
            smiles_to_encode = smiles_list

        if not smiles_to_encode:
            embeddings = {}
        else:
            embeddings = self._encode_smiles(smiles_to_encode)

        if skip_existing:
            all_embeddings = {**smiles_encoded, **embeddings}
            embeddings = {s: all_embeddings[s] for s in smiles_list}

        # Update instance embeddings
        if len(embeddings) > 0:
            self.embeddings.update(embeddings)

        if update_cache:
            self.save()

        # Return single embedding if input was a single SMILES
        if isinstance(smiles, str):
            return embeddings[smiles]
        return embeddings

    def _encode_smiles(self, smiles_list: List[str]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """ Internal method to encode SMILES based on the configured embeddings_type. """
        if self.embeddings_type == "fingerprint":
            return self._encode_smiles_as_fingerprints(smiles_list)
        elif self.embeddings_type == "rdkit_descriptors":
            return self._encode_rdkit_descriptors(smiles_list)
        elif self.embeddings_type == "transformer":
            return self._encode_with_transformer(smiles_list)
        else:
            raise ValueError(f"Unsupported embeddings_type: {self.embeddings_type}, must be one of 'fingerprint', 'rdkit_descriptors', or 'transformer'.")

    def _encode_smiles_as_fingerprints(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """ Encode SMILES as Morgan fingerprints. """
        morgan_fpgen = self._morgan_fpgen
        if morgan_fpgen is None:
            morgan_fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius,
                fpSize=self.fp_size,
                includeChirality=True,
            )
            self._morgan_fpgen = morgan_fpgen
        fingerprints = {}
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            else:
                fp = morgan_fpgen.GetFingerprintAsNumPy(mol).astype(np.float32)
                fingerprints[smiles] = fp
        return fingerprints
    
    def _encode_rdkit_descriptors(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """ Encode SMILES as RDKit molecular descriptors. """
        descriptors = {}
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            else:
                desc_dict = self.get_mol_descriptors(mol)

                # Clip all descriptor values to be finite
                for name, val in desc_dict.items():
                    if val > 1e20 or val < -1e20 or val is None or np.isnan(val) or np.isinf(val):
                        desc_dict[name] = -1
                
                desc_array = np.array(list(desc_dict.values()), dtype=np.float32).flatten()
                descriptors[smiles] = desc_array
        return descriptors

    def _encode_with_transformer(self, smiles_list: List[str]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """ Encode SMILES using transformer model. """
        return self.encode_with_transformer(
            strings=smiles_list,
            tokenizer=self.tokenizer,
            model=self.model,
            pretrained_model=self.pretrained_model,
            batch_size=self.batch_size,
            device=self.device,
            pooling=self.pooling,
            return_tensors=self.return_tensors,
            return_dict=True,
        )

    @staticmethod
    def encode_smiles_as_fingerprints(
        smiles: Union[str, List[str]],
        morgan_fpgen: Optional[FingerprintGenerator64] = None,
        radius: int = config.morgan_radius,
        fp_size: int = config.fingerprint_size,
    ) -> Dict[str, np.ndarray]:
        """ Static method to get the Morgan fingerprint of molecules.
        
        Args:
            smiles: The SMILES string(s) of the molecule(s)
            morgan_fpgen: The Morgan fingerprint generator
            radius: Radius for Morgan fingerprints
            fp_size: Size of the Morgan fingerprints

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping SMILES to fingerprints
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
        elif isinstance(smiles, list):
            smiles_list = smiles
        else:
            raise ValueError("Input smiles must be a string or a list of strings.")

        if morgan_fpgen is None:
            morgan_fpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
                radius=radius,
                fpSize=fp_size,
                includeChirality=True,
            )

        fingerprints = {}
        for smiles_str in smiles_list:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles_str}")
            else:
                fp = morgan_fpgen.GetFingerprintAsNumPy(mol).astype(np.float32)
                fingerprints[smiles_str] = fp

        return fingerprints

    def get_mol_descriptors(
        self,
        mol: Union[str, Chem.Mol],
        missing: Any = np.nan,
    ) -> dict:
        """ Calculate the full list of descriptors for a molecule.
        
        Args:
            mol (Union[str, Chem.Mol]): The molecule as a SMILES string or an RDKit Mol object.
            missing (Any): Value to use if a descriptor cannot be calculated.
            
        Returns:
            dict: A dictionary mapping descriptor names to their calculated values.
        """
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        res = {}
        descriptors_list = self.get_descriptor_names()
        for name, fn in Descriptors._descList:
            if name not in descriptors_list:
                continue
            # Some of the descriptor fucntions can throw errors if they fail,
            # catch those here:
            try:
                val = fn(mol)
            except:
                # And set the descriptor value to whatever `missing` is
                val = missing
            res[name] = val
        return res

RELEVANT_RDKIT_DESCRIPTORS = [
    "VSA_EState3",
    "FractionCSP3",
    "AvgIpc",
    "BCUT2D_LOGPHI",
    "Chi4n",
    "VSA_EState2",
    "MaxAbsEStateIndex",
    "BCUT2D_MRLOW",
    "EState_VSA3",
    "VSA_EState4",
    "SMR_VSA3",
    "SMR_VSA7",
    "BCUT2D_LOGPLOW",
    "TPSA",
    "EState_VSA5",
    "SMR_VSA1",
    "BCUT2D_CHGLO",
    "fr_ArN",
    "EState_VSA9",
    "BCUT2D_CHGHI",
    "qed",
    "NumUnspecifiedAtomStereoCenters",
    "VSA_EState10",
    "BCUT2D_MRHI",
    "SMR_VSA6",
    "SMR_VSA10",
    "MinAbsEStateIndex",
    "PEOE_VSA10",
    "SPS",
    "SlogP_VSA3",
    "PEOE_VSA9",
    "FpDensityMorgan2",
    "BalabanJ",
    "SlogP_VSA2",
    "SlogP_VSA1",
    "FpDensityMorgan3",
    "PEOE_VSA8",
    "MaxEStateIndex",
    "fr_NH0",
    "VSA_EState8",
    "VSA_EState6",
    "VSA_EState5",
    "EState_VSA1",
    "SMR_VSA5",
    "PEOE_VSA11",
    "BertzCT",
    "Chi4v",
    "Kappa3",
    "MaxPartialCharge",
    "fr_bicyclic",
    "VSA_EState1",
    "MinEStateIndex",
    "PEOE_VSA2",
    "MolLogP",
    "PEOE_VSA7",
    "EState_VSA2",
    "FpDensityMorgan1",
    "SMR_VSA9",
    "EState_VSA8",
    "EState_VSA4",
    "EState_VSA7",
    "PEOE_VSA1",
    "EState_VSA6",
    "fr_NH2",
    "BCUT2D_MWLOW",
    "VSA_EState7",
    "SlogP_VSA4",
    "PEOE_VSA6",
    "VSA_EState9",
    "Chi3n",
    "SlogP_VSA10",
    "Kappa2",
    "fr_aniline",
    "MolMR",
    "fr_ether",
    "NumHAcceptors",
    "RingCount",
    "PEOE_VSA12",
    "MaxAbsPartialCharge",
    "Chi3v",
    "HallKierAlpha",
    "MinAbsPartialCharge",
    "PEOE_VSA3",
    "NumAtomStereoCenters",
    "fr_piperdine",
    "Ipc",
    "Chi2v",
    "SlogP_VSA5",
    "MinPartialCharge",
    "NumRotatableBonds",
    "fr_term_acetylene",
    "fr_NH1",
    "Chi1n",
    "NumHeteroatoms",
    "SlogP_VSA6",
    "Kappa1",
    "NumAromaticHeterocycles",
    "fr_Ar_N",
    "Phi",
    "SMR_VSA4",
    "MolWt",
    "fr_Ndealkylation2",
    "SlogP_VSA11",
    "NumHeterocycles",
    "PEOE_VSA4",
    "fr_unbrch_alkane",
    "EState_VSA10",
    "HeavyAtomMolWt",
    "NumHDonors",
    "Chi0",
    "BCUT2D_MWHI",
    "Chi2n",
    "NOCount",
    "SlogP_VSA8",
    "NumSaturatedRings",
    "LabuteASA",
    "fr_piperzine",
    "NumSaturatedHeterocycles",
    "Chi1",
    "Chi1v",
    "NHOHCount",
    "ExactMolWt",
    "NumAliphaticHeterocycles",
    "PEOE_VSA14",
    "NumValenceElectrons",
    "PEOE_VSA13",
    "Chi0v",
    "fr_Ar_NH",
    "NumAromaticRings",
    "EState_VSA11",
    "fr_halogen",
    "NumAmideBonds",
    "SlogP_VSA7",
    "PEOE_VSA5",
    "NumSaturatedCarbocycles",
    "SlogP_VSA12",
    "fr_imide",
    "fr_Al_OH",
    "NumAliphaticCarbocycles",
    "NumAromaticCarbocycles",
    "NumSpiroAtoms",
    "fr_amide",
    "Chi0n",
    "fr_Nhpyrrole",
    "fr_allylic_oxid",
    "NumAliphaticRings",
    "fr_benzene",
    "fr_alkyl_halide",
    "fr_imidazole",
    "fr_para_hydroxylation",
    "fr_C_O_noCOO",
    "fr_methoxy",
    "HeavyAtomCount",
    "fr_oxazole",
    "SMR_VSA2",
    "fr_HOCCN",
    "fr_thiazole",
    "NumBridgeheadAtoms",
    "fr_ester",
    "fr_hdrzone",
    "fr_ketone",
    "fr_furan",
    "fr_COO2",
    "fr_Ndealkylation1",
    "fr_priamide",
    "fr_sulfonamd",
]