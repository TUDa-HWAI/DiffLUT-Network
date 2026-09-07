import torch
import numpy as np
import os
import json


class TabularThermometer:
    """Feature-wise quantile thermometer encoding for tabular arrays."""

    def __init__(self, num_bits):
        self.num_bits = int(num_bits)
        self.thresholds = None

    def fit(self, x):
        quantiles = np.linspace(0, 1, self.num_bits + 2)[1:-1]
        self.thresholds = np.quantile(x, quantiles, axis=0).T.astype(np.float32)
        return self

    def binarize(self, x):
        if self.thresholds is None:
            raise RuntimeError("TabularThermometer must be fitted before binarize()")
        return (x[:, :, np.newaxis] >= self.thresholds[np.newaxis]).astype(np.uint8)


class ArrayDataset(torch.utils.data.Dataset):
    """
    Dataset for Jet Substructure Classification.
    
    Data is expected to be pre-binarized using DistributiveThermometer encoding.
    Expected data format: 
    - data: [N, input_dim] float array (binarized features)
    - labels: [N] long/int array
    """
    def __init__(self, data_path, label_path, transform=None):
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Processed dataset file {data_path} was not found."
            )
        
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        # Convert to torch tensor on-the-fly (copies data to avoid write-protection warnings)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        if self.transform:
            x = self.transform(x)
        return x, y

    @staticmethod
    def get_input_dim(data_dir):
        """
        Read the input dimension from metadata, or infer from data file.
        """
        metadata_path = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata['input_dim']
        
        # Fallback: infer from saved data
        data_path = os.path.join(data_dir, 'train_data.npy')
        if os.path.exists(data_path):
            data = np.load(data_path, mmap_mode='r')
            return data.shape[1]
        
        raise FileNotFoundError(f"No metadata or training data found in {data_dir}")
