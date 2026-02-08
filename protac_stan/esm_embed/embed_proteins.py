import argparse
import hashlib
import os
import pickle
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
    return seq.strip().replace('U', 'C').replace('O', 'K').replace('B', 'D').replace('Z', 'E').replace('J', 'L')


def get_protein_key(uniprot_id, sequence: str) -> str:
    if pd.notna(uniprot_id) and str(uniprot_id).strip() not in ['None', '', 'nan']:
        return str(uniprot_id).strip()
    return hashlib.md5(sanitize_sequence(sequence).encode('utf-8')).hexdigest()


def load_data(args) -> pd.DataFrame:
    if args.custom_dataset_csv:
        print(f"Loading custom CSV: {args.custom_dataset_csv}")
        return pd.read_csv(args.custom_dataset_csv)
    return None


def collect_proteins(args) -> dict:
    protein_map = {}
    
    if args.custom_dataset_csv:
        df = load_data(args)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning proteins"):
            if pd.notna(row.get('POI_Sequence')):
                key = get_protein_key(row.get('POI_UniProt'), row['POI_Sequence'])
                protein_map[key] = sanitize_sequence(row['POI_Sequence'])
            if pd.notna(row.get('Ligase_Sequence')):
                key = get_protein_key(row.get('Ligase_UniProt'), row['Ligase_Sequence'])
                protein_map[key] = sanitize_sequence(row['Ligase_Sequence'])
    else:
        for config in CONFIGS:
            print(f"Loading '{config}' from {DATASET_NAME}...")
            try:
                ds = load_dataset(DATASET_NAME, config, split='train', token=True)
                df = ds.to_pandas()
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scanning {config}"):
                    if pd.notna(row.get('POI_Sequence')):
                        key = get_protein_key(row.get('POI_UniProt'), row['POI_Sequence'])
                        protein_map[key] = sanitize_sequence(row['POI_Sequence'])
                    if pd.notna(row.get('Ligase_Sequence')):
                        key = get_protein_key(row.get('Ligase_UniProt'), row['Ligase_Sequence'])
                        protein_map[key] = sanitize_sequence(row['Ligase_Sequence'])
            except Exception as e:
                print(f"  Failed: {e}")
    
    return protein_map


def main():
    parser = argparse.ArgumentParser(description="Generate ESM protein embeddings")
    parser.add_argument('--custom_dataset_csv', type=str, default=None,
                        help='Path to custom CSV (overrides HuggingFace)')
    parser.add_argument('--output_dir', type=str, default='../data/custom')
    parser.add_argument('--model_dir', type=str, default='./model')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    pmap_path = os.path.join(args.output_dir, 'p_map.pkl')
    embed_path = os.path.join(args.output_dir, 'esm_s_map.pkl')
    
    p_map = pickle.load(open(pmap_path, 'rb')) if os.path.exists(pmap_path) else {}
    embed_map = pickle.load(open(embed_path, 'rb')) if os.path.exists(embed_path) else {}
    
    print("Collecting proteins...")
    new_proteins = collect_proteins(args)
    p_map.update(new_proteins)
    
    with open(pmap_path, 'wb') as f:
        pickle.dump(p_map, f)
    print(f"Updated p_map.pkl (Total: {len(p_map)})")
    
    missing_ids = [uid for uid in new_proteins if uid not in embed_map]
    print(f"Missing embeddings: {len(missing_ids)}")
    
    if not missing_ids:
        print("All proteins embedded. Done.")
        return
    
    print("Loading ESM-2 Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = models.EvolutionaryScaleModeling(args.model_dir, model="ESM-2-650M", readout="mean")
    weights = torch.load(os.path.join(args.model_dir, "esm_650m_s.pth"), map_location="cpu")
    esm_model.load_state_dict(weights)
    esm_model = esm_model.to(device).eval()
    
    print(f"Generating {len(missing_ids)} embeddings...")
    for uid in tqdm(missing_ids):
        seq = new_proteins[uid][:1022]
        try:
            protein = data.Protein.from_sequence(seq)
            graph = data.Protein.pack([protein]).to(device)
            with torch.no_grad():
                output = esm_model(graph, graph.node_feature.float())
                embed_map[uid] = output['graph_feature'].cpu().detach().numpy()[0]
        except Exception as e:
            print(f"FAILED {uid}: {e}")
    
    with open(embed_path, 'wb') as f:
        pickle.dump(embed_map, f)
    print(f"Saved esm_s_map.pkl ({len(embed_map)} embeddings)")


if __name__ == "__main__":
    main()
