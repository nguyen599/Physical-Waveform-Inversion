import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
from joblib import Parallel, delayed
from config import cfg

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        mode = "train",
        inv_scale=False,
    ):
        self.cfg = cfg
        self.mode = mode
        self.inv_scale = inv_scale

        self.data, self.labels, self.records = self.load_metadata()

    def _load_single_file(self, row_dict, mmap_mode="r"):
        """Load a single file pair (data and label)"""
        try:
            # Hacky way to get exact file name
            p1 = os.path.join(f"{cfg.data_path}open-wfi-1/openfwi_float16_1/", row_dict["data_fpath"])
            p2 = os.path.join(f"{cfg.data_path}open-wfi-1/openfwi_float16_1/", row_dict["data_fpath"].split("/")[0], "*", row_dict["data_fpath"].split("/")[-1])
            p3 = os.path.join(f"{cfg.data_path}open-wfi-2/openfwi_float16_2/", row_dict["data_fpath"])
            p4 = os.path.join(f"{cfg.data_path}open-wfi-2/openfwi_float16_2/", row_dict["data_fpath"].split("/")[0], "*", row_dict["data_fpath"].split("/")[-1])
            farr = glob.glob(p1) + glob.glob(p2) + glob.glob(p3) + glob.glob(p4)

            # Map to lbl fpath
            farr = farr[0]
            flbl = farr.replace('seis', 'vel').replace('data', 'model')

            # Load with random mmap decision
            current_mmap_mode = None if random.random() < 0.00 else mmap_mode
            arr = np.load(farr, mmap_mode=current_mmap_mode)
            lbl = np.load(flbl, mmap_mode=current_mmap_mode)

            return arr, lbl, row_dict["dataset"]
        except Exception as e:
            print(f"Error loading file {row_dict.get('data_fpath', 'unknown')}: {e}")
            return None, None, None

    def load_metadata(self):
        # Select rows
        df = pd.read_csv("/kaggle/input/openfwi-preprocessed-72x72/folds.csv")
        if self.cfg.subsample is not None:
            df = df.groupby(["dataset", "fold"]).head(self.cfg.subsample)
        if self.mode == "train":
            df = df[df["fold"] != 0]
        else:
            df = df[df["fold"] == 0]

        # Convert dataframe rows to list of dictionaries for parallel processing
        row_dicts = [row.to_dict() for idx, row in df.iterrows()]

        # Determine number of jobs (use all available cores, but cap at reasonable number)
        n_jobs = min(len(row_dicts), os.cpu_count() or 1, 8)  # Cap at 8 to avoid overwhelming I/O
        n_jobs = 36

        # Use joblib to load files in parallel with progress bar
        print(f"Loading {len(row_dicts)} files using {n_jobs} parallel workers...")

        # Create progress bar that works with parallel processing
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self._load_single_file)(row_dict)
            for row_dict in tqdm(row_dicts, disable=self.cfg.local_rank != 0, desc="Loading files")
        )

        # Separate results and filter out failed loads
        data = []
        labels = []
        records = []

        failed_count = 0
        for arr, lbl, dataset in results:
            if arr is not None and lbl is not None:
                data.append(arr)
                labels.append(lbl)
                records.append(dataset)
            else:
                failed_count += 1

        if failed_count > 0:
            print(f"Warning: {failed_count} files failed to load")

        print(f"Successfully loaded {len(data)} files")
        return data, labels, records

    def __getitem__(self, idx):
        row_idx = idx // 500
        col_idx = idx % 500
        d = self.records[row_idx]

        x = self.data[row_idx][col_idx, ...]
        if self.inv_scale:
            y = (self.labels[row_idx][col_idx, ...] - 3000) / 1500
        else:
            y = self.labels[row_idx][col_idx, ...]

        # Augs
        if self.mode == "train":
            # Temporal flip
            if np.random.random() < 0.5:
                x = x[::-1, :, ::-1]
                y = y[..., ::-1]

        x = x.copy()
        y = y.copy()

        return x, y, d

    def __len__(self):
        return len(self.records) * 500