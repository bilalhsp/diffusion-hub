import torch
import h5py
import time
import numpy as np
from torch.utils.data import Dataset

import torch, random
from pathlib import Path
import numpy as np

def compute_spectrogram_stats(block_indices, root_pattern):
    """Computes Z-score stats for each block individually."""
    all_stats = {}
    total_sum = 0
    total_sq_sum = 0
    total_count = 0
    print("Calculating per-block statistics...")
    for b_idx in block_indices:
        path = Path(root_pattern.format(b_idx))
        spec = np.load(path / "Spec_continuous.npy", mmap_mode='r')
        # spec shape: (channels, time) = (80 mel bins, time)
        # We sum across the time dimension (axis 1)
        total_count += spec.shape[1]
        s = np.sum(spec, axis=1, keepdims=True)
        sq_s = np.sum(spec**2, axis=1, keepdims=True)
        total_sum += s
        total_sq_sum += sq_s
    avg_mean = total_sum / total_count
    avg_var = (total_sq_sum / total_count) - (avg_mean ** 2)
    avg_std = np.sqrt(np.maximum(avg_var, 0)) # Clip at 0 for precision errors
    all_stats = {
        'spec_mean': avg_mean.astype(np.float32),
        'spec_std': avg_std.astype(np.float32)
    }
    return all_stats

class NeuralSpeechDataset(Dataset):
    def __init__(self, root_dir, block_id, T_segment=2.0, mode='train'):

        self.root_dir = root_dir
        self.block_id = block_id
        self.path = Path(root_dir) / f"processed_block_{block_id:03d}_full"

        self.T_segment = T_segment
        self.mode = mode.lower()
        meta = np.load(self.path / "metadata.npy", allow_pickle=True).item()
        self.fs_eeg, self.fs_spec = meta['fs_eeg'], meta['fs_spec']
        # Filter valid_ranges to only include the split indices (train or val)
        # self.ranges = [meta['valid_ranges'][i] for i in indices]

        # split each trial into 2 segments
        ranges = []
        for strt_time, _ in meta['valid_ranges']:           # total duration ~= 12 seconds
            ranges.append((strt_time, strt_time + 6.0))
            ranges.append((strt_time+5.0, strt_time+11.0))

        self.ranges = np.array(ranges)
        self.eeg_data = np.load(self.path / "sEEG_continuous.npy")
        self.spec_data = np.load(self.path / "Spec_continuous.npy")


        if self.mode == 'train':
            np.random.shuffle(self.ranges)  # Shuffle the order of segments for training
        else:
            total_segments = len(self.ranges)
            indices = np.arange(total_segments)
            rng = np.random.RandomState(42)
            val_indices = rng.choice(total_segments, size=total_segments//2, replace=False)
            if self.mode == 'val':
                self.ranges = self.ranges[val_indices]
            elif self.mode == 'test':
                test_mask = np.ones(total_segments, dtype=bool)
                test_mask[val_indices] = False
                self.ranges = self.ranges[test_mask]        


    def __len__(self):
        return len(self.ranges)

    def __getitem__(self, idx):

        start_time, end_time = self.ranges[idx]
        x = self.eeg_data[:, int(start_time * self.fs_eeg) : int(end_time * self.fs_eeg)].copy()  # copy to avoid mmap issues
        y = self.spec_data[:, int(start_time * self.fs_spec) : int(end_time * self.fs_spec)].copy()  # copy to avoid mmap issues

        if self.mode == 'train': ## do augmentations
            mask = np.random.rand(x.shape[0]) > 0.1
            x = x * mask[:, None]  # broadcast over time
            if random.random() < 0.5:
                x += np.random.normal(0, 0.316, x.shape) ## 0.316 is std dev
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    

if __name__ == "__main__":
    config = {
        'root_dir': '/scratch/gilbreth/ahmedb/data/nz/labelled/processed_blocks/',
        'block_id': 5,
        'T_segment': 2.0,
        'mode': 'train'
    }


    print(f"creating dataset with config: {config}")
    ds = NeuralSpeechDataset(**config)

    print(f"Measuring dataloader performance...")
    times = []
    for _ in range(10):  # warmup
        for idx in range(len(ds)):
            start = time.perf_counter()
            example = ds[idx]
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    print(f"avg: {np.mean(times):.4f}s")
    print(f"min: {np.min(times):.4f}s")
    print(f"max: {np.max(times):.4f}s")