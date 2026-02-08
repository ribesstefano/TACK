import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch
from data import PROTACData


def collate_fn(data_list):
    """Collate function for batching PROTAC data with PyG graphs."""
    batch = {
        'protac': Batch.from_data_list([item['protac'] for item in data_list]),
        'e3_ligase': torch.stack([item['e3_ligase'] for item in data_list]),
        'poi': torch.stack([item['poi'] for item in data_list]),
    }
    labels = [item['label'] for item in data_list]
    batch['label'] = torch.stack(labels) if labels[0] is not None else None
    return batch


class PROTACDataset(Dataset):
    """Dataset combining PROTAC graphs with protein embeddings and labels."""
    
    def __init__(self, protac, e3_ligase, poi, label=None):
        self.protac = protac
        self.e3_ligase = e3_ligase
        self.poi = poi
        self.label = label

    def __len__(self):
        return len(self.protac)

    def __getitem__(self, index):
        return {
            'protac': self.protac[index],
            'e3_ligase': self.e3_ligase[index],
            'poi': self.poi[index],
            'label': self.label[index] if self.label is not None else None
        }


def PROTACLoader(root='data/custom', name='custom', batch_size=32, collate_fn=collate_fn,
                 train_ratio=0.8, train_indices=None, test_indices=None):
    """
    Create DataLoaders for PROTAC training and evaluation.
    
    Args:
        root: Directory containing processed data
        name: Dataset name (e.g., 'custom_dc50', 'held_out_dc50')
        batch_size: Batch size for DataLoaders
        collate_fn: Collation function for batching
        train_ratio: Train/test split ratio (ignored if indices provided)
        train_indices: Explicit training indices for cross-validation
        test_indices: Explicit test indices for cross-validation
    
    Returns:
        (train_loader, test_loader) tuple
    """
    # Load processed graph data and embeddings
    protac = PROTACData(root, name=name)
    processed_path = f'{root}/processed/{name}'
    
    e3_ligase = torch.load(f'{processed_path}/e3_ligase.pt')
    poi = torch.load(f'{processed_path}/poi.pt')
    try:
        label = torch.load(f'{processed_path}/label.pt')
    except FileNotFoundError:
        label = None

    full_dataset = PROTACDataset(protac, e3_ligase, poi, label)

    # Cross-validation mode: use explicit indices
    if train_indices is not None and test_indices is not None:
        train_loader = DataLoader(
            Subset(full_dataset, train_indices),
            batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            Subset(full_dataset, test_indices),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        return train_loader, test_loader

    # Inference mode: return full dataset as test loader
    if train_ratio == 0.0:
        print(f'Test Dataset:\nTotal size:  {len(full_dataset)}')
        return None, DataLoader(full_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Random split mode (legacy)
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    print(f'Total: {len(full_dataset)} | Train: {train_size} | Test: {test_size}')
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    # Remove overlapping SMILES from test set
    train_smiles = {data['protac'].smiles for data in train_dataset}
    test_dataset = [d for d in test_dataset if d['protac'].smiles not in train_smiles]
    print(f'After dedup: Train: {len(train_dataset)} | Test: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, test_loader
