import argparse
import hashlib
import os
import pickle
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torchdrug import data, models
from tqdm import tqdm

DATASET_NAME = "ailab-bio/TACK"
CONFIGS = ['default', 'Dmax', 'DC50', 'multitask']


def sanitize_sequence(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    # Chained replace is efficient enough for this string length
    return seq.strip().replace('U', 'C').replace('O', 'K').replace('B', 'D').replace('Z', 'E').replace('J', 'L')


def get_protein_key(uniprot_id, sequence: str) -> str:
    if pd.notna(uniprot_id) and str(uniprot_id).strip() not in ['None', '', 'nan']:
        return str(uniprot_id).strip()
    return hashlib.md5(sequence.encode('utf-8')).hexdigest()


def collect_proteins(args) -> tuple:
    """
    Scans datasets and builds routing maps.
    Returns:
        key_to_sanitized:  {UniProt-or-MD5 -> sanitized_sequence}
        orig_to_sanitized: {original_sequence -> sanitized_sequence, 
                            sanitized_sequence -> sanitized_sequence}
    """
    key_to_sanitized = {}
    orig_to_sanitized = {}

    def extract_from_df(df: pd.DataFrame, desc: str):
        for prefix in ['POI', 'Ligase']:
            seq_col, id_col = f'{prefix}_Sequence', f'{prefix}_UniProt'
            if seq_col not in df.columns:
                continue
            
            # Drop empty sequences at the pandas level for faster iteration
            valid_rows = df.dropna(subset=[seq_col])
            
            for _, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc=f"{desc} ({prefix})"):
                orig_seq = row[seq_col]
                uniprot_id = row.get(id_col)
                
                sanitized_seq = sanitize_sequence(orig_seq)
                key = get_protein_key(uniprot_id, sanitized_seq)

                key_to_sanitized[key] = sanitized_seq
                orig_to_sanitized[orig_seq] = sanitized_seq
                orig_to_sanitized[sanitized_seq] = sanitized_seq

    if args.custom_dataset_csv:
        print(f"Loading custom CSV: {args.custom_dataset_csv}")
        df = pd.read_csv(args.custom_dataset_csv)
        extract_from_df(df, "Scanning custom CSV")
    else:
        for config in CONFIGS:
            print(f"Loading '{config}' from {DATASET_NAME}...")
            try:
                ds = load_dataset(DATASET_NAME, config, split='train', token=True)
                extract_from_df(ds.to_pandas(), f"Scanning {config}")
            except Exception as e:
                print(f"  Failed: {e}")

    return key_to_sanitized, orig_to_sanitized


def main():
    parser = argparse.ArgumentParser(description="Generate ESM protein embeddings")
    parser.add_argument('--custom_dataset_csv', type=str, default=None,
                        help='Path to custom CSV (overrides HuggingFace)')
    parser.add_argument('--output_dir', type=str, default='../data/custom')
    parser.add_argument('--model_dir', type=str, default='./model')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    pmap_path = os.path.join(args.output_dir, 'p_map.pkl')
    pmap_npz_path = os.path.join(args.output_dir, 'p_map.npz')
    embed_path = os.path.join(args.output_dir, 'esm_s_map.pkl')
    embed_npz_path = os.path.join(args.output_dir, 'esm_s_map.npz')
    seq_embed_path = os.path.join(args.output_dir, 'seq_esm_s_map.pkl')
    seq_embed_npz_path = os.path.join(args.output_dir, 'seq_esm_s_map.npz')

    # Load existing caches
    p_map = pickle.load(open(pmap_path, 'rb')) if os.path.exists(pmap_path) else {}
    embed_map = pickle.load(open(embed_path, 'rb')) if os.path.exists(embed_path) else {}
    seq_embed_map = pickle.load(open(seq_embed_path, 'rb')) if os.path.exists(seq_embed_path) else {}

    print("Collecting proteins...")
    key_to_sanitized, orig_to_sanitized = collect_proteins(args)
    
    # Update and save the simple protein mapping
    p_map.update(key_to_sanitized)
    with open(pmap_path, 'wb') as f:
        pickle.dump(p_map, f)
    np.savez(pmap_npz_path, **{k: np.array(v) for k, v in p_map.items()})
    print(f"Updated p_map.pkl / p_map.npz (Total keys: {len(p_map)})")

    # Determine strictly which unique SANITIZED sequences need to be embedded
    unique_sanitized_seqs = set(key_to_sanitized.values())
    missing_seqs = [seq for seq in unique_sanitized_seqs if seq not in seq_embed_map]
    
    print(f"Missing sequence embeddings to generate: {len(missing_seqs)}")

    if missing_seqs:
        print("Loading ESM-2 Model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        esm_model = models.EvolutionaryScaleModeling(args.model_dir, model="ESM-2-650M", readout="mean")
        weights = torch.load(os.path.join(args.model_dir, "esm_650m_s.pth"), map_location="cpu")
        esm_model.load_state_dict(weights)
        esm_model = esm_model.to(device).eval()

        def embed_sequence(seq: str) -> np.ndarray:
            protein = data.Protein.from_sequence(seq[:1022])
            graph = data.Protein.pack([protein]).to(device)
            with torch.no_grad():
                output = esm_model(graph, graph.node_feature.float())
            return output['graph_feature'].cpu().detach().numpy()[0]

        print("Generating embeddings...")
        for seq in tqdm(missing_seqs):
            try:
                # Store against the sanitized sequence as our baseline
                seq_embed_map[seq] = embed_sequence(seq)
            except Exception as e:
                print(f"FAILED seq({seq[:20]}...): {e}")

    # --- Fan Out Logic ---
    # 1. Populate embed_map based on key -> sanitized_seq
    for key, san_seq in key_to_sanitized.items():
        if san_seq in seq_embed_map:
            embed_map[key] = seq_embed_map[san_seq]

    # 2. Populate seq_embed_map for both original and sanitized sequences
    for orig_seq, san_seq in orig_to_sanitized.items():
        if san_seq in seq_embed_map:
            seq_embed_map[orig_seq] = seq_embed_map[san_seq]

    # Save embed_map (Key-based)
    with open(embed_path, 'wb') as f:
        pickle.dump(embed_map, f)
    np.savez(embed_npz_path, **embed_map)
    print(f"Saved esm_s_map.pkl / esm_s_map.npz ({len(embed_map)} keys mapped)")

    # Save seq_embed_map (Sequence-based)
    # npz uses MD5(seq) as the array name since sequences are too long for zip entry names
    with open(seq_embed_path, 'wb') as f:
        pickle.dump(seq_embed_map, f)
    # seq_npz_data = {hashlib.md5(seq.encode()).hexdigest(): emb for seq, emb in seq_embed_map.items()}
    # np.savez(seq_embed_npz_path, **seq_npz_data)
    np.savez(seq_embed_npz_path, **seq_embed_map)
    print(f"Saved seq_esm_s_map.pkl / seq_esm_s_map.npz ({len(seq_embed_map)} sequence keys mapped)")
    print(f"Point to the npz file '{embed_npz_path}' in data configs to use them for training TACK models.")


if __name__ == "__main__":
    main()