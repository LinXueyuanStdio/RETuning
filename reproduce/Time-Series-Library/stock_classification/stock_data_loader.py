"""
Stock Classification Data Loader for Time-Series-Library.

Loads NPZ datasets created by build_stock_classification_dataset.py
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class StockClassificationDataset(Dataset):
    """
    Dataset for stock overnight rate classification.

    Loads pre-built NPZ files containing:
    - x: [N, seq_len, 1] - overnight rate sequences
    - y: [N] - labels (0=down, 1=hold, 2=up)
    """

    def __init__(
        self,
        args,
        root_path: str,
        flag: str = "TRAIN",
        mode: int = 1,
        seq_len: int = 20,
    ):
        """
        Args:
            args: argument namespace
            root_path: root directory containing dataset folders
            flag: "TRAIN" or "TEST"
            mode: 1 or 2 (data split mode)
            seq_len: sequence length
        """
        self.args = args
        self.root_path = root_path
        self.flag = flag.upper()
        self.mode = mode
        self.seq_len = seq_len

        # Load dataset
        dataset_name = f"Stock_mode{mode}_sl{seq_len}"
        dataset_dir = os.path.join(root_path, dataset_name)

        npz_path = os.path.join(dataset_dir, f"{self.flag}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Dataset not found at {npz_path}. "
                f"Please run build_stock_classification_dataset.py first."
            )

        data = np.load(npz_path)
        self.x = data["x"].astype(np.float32)  # [N, seq_len, 1]
        self.y = data["y"].astype(np.int64)    # [N]

        # Load metadata
        meta_path = os.path.join(dataset_dir, f"{self.flag.lower()}_meta.csv")
        if os.path.exists(meta_path):
            self.meta_df = pd.read_csv(meta_path)
        else:
            self.meta_df = None

        # Normalize features (fit on train, apply to both)
        self.scaler = StandardScaler()
        if self.flag == "TRAIN":
            # Fit scaler on training data
            x_flat = self.x.reshape(-1, 1)
            self.scaler.fit(x_flat)
        else:
            # Load scaler parameters from training data
            train_npz_path = os.path.join(dataset_dir, "TRAIN.npz")
            if os.path.exists(train_npz_path):
                train_data = np.load(train_npz_path)
                train_x = train_data["x"].astype(np.float32)
                x_flat = train_x.reshape(-1, 1)
                self.scaler.fit(x_flat)

        # Apply normalization
        original_shape = self.x.shape
        self.x = self.scaler.transform(self.x.reshape(-1, 1)).reshape(original_shape)

        # Class information
        self.num_classes = 3
        self.class_names = ["down", "hold", "up"]
        self.feature_dim = 1
        self.max_seq_len = self.seq_len

        # Create feature DataFrame for compatibility
        self.feature_df = pd.DataFrame({"overnight_rate": [0.0]})

        print(f"Loaded {self.flag} dataset: {len(self.x)} samples, "
              f"seq_len={self.seq_len}, feature_dim={self.feature_dim}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        Returns:
            x: [seq_len, feature_dim] tensor
            y: scalar label
        """
        x = torch.from_numpy(self.x[idx])  # [seq_len, 1]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

    def get_meta(self, idx):
        """Get metadata for a sample."""
        if self.meta_df is not None:
            return self.meta_df.iloc[idx].to_dict()
        return None


def collate_fn_stock(data, max_len=None):
    """
    Collate function for stock classification.

    Args:
        data: list of (x, y) tuples
        max_len: maximum sequence length (unused, kept for compatibility)

    Returns:
        X: [batch_size, seq_len, feature_dim] tensor
        targets: [batch_size] tensor
        padding_masks: [batch_size, seq_len] tensor (all ones, no padding)
    """
    batch_size = len(data)
    features, labels = zip(*data)

    # Stack features
    X = torch.stack(features, dim=0)  # [batch_size, seq_len, feature_dim]
    seq_len = X.shape[1]

    # Stack labels
    targets = torch.stack(labels, dim=0)  # [batch_size]

    # Create padding mask (all ones since sequences are fixed length)
    padding_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)

    return X, targets, padding_masks


def data_provider_stock(args, flag):
    """
    Data provider for stock classification.

    Args:
        args: argument namespace with:
            - root_path: dataset root directory
            - mode: 1 or 2
            - seq_len: sequence length
            - batch_size: batch size
            - num_workers: number of data loading workers
        flag: "TRAIN" or "TEST"

    Returns:
        data_set: StockClassificationDataset
        data_loader: DataLoader
    """
    from torch.utils.data import DataLoader

    data_set = StockClassificationDataset(
        args=args,
        root_path=args.root_path,
        flag=flag,
        mode=args.mode,
        seq_len=args.seq_len,
    )

    shuffle = (flag.upper() == "TRAIN")

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=lambda x: collate_fn_stock(x, max_len=args.seq_len)
    )

    return data_set, data_loader
