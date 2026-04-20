import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

import torch, random
from pathlib import Path
import numpy as np

# local
from .factory import register_dataset

# data in: /srv/share/NZ0_spectrograms/
# Adding code to this repo for completeness, 
# Data is on phocion, not on Gilbreth. Run on Phocion.

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

@register_dataset("nz_label")
def create_combined_dataset(root_dir, block_ids, trial_duration=4.0, mode='train'):
    if len(block_ids) ==1:
        return NeuralSpeechDataset(root_dir, block_ids[0], trial_duration, mode)
    datasets = [NeuralSpeechDataset(root_dir, b_id, trial_duration, mode) for b_id in block_ids]
    combined = torch.utils.data.ConcatDataset(datasets)
    return combined


class NeuralSpeechDataset(Dataset):
    def __init__(self, root_dir, block_id, trial_duration=4.0, mode='train'):

        self.root_dir = root_dir
        self.block_id = block_id
        self.path = Path(root_dir) / f"processed_block_{block_id:03d}_full"

        self.max_trial_duration = 6
        assert trial_duration > 0 and trial_duration <= self.max_trial_duration, "Trial duration must be between 0 and 6 seconds"
        self.trial_duration = trial_duration
        
        self.mode = mode.lower()
        meta = np.load(self.path / "metadata.npy", allow_pickle=True).item()
        self.fs_eeg, self.fs_spec = meta['fs_eeg'], meta['fs_spec']
        # Filter valid_ranges to only include the split indices (train or val)
        # self.ranges = [meta['valid_ranges'][i] for i in indices]

        ranges = [meta['valid_ranges'][i] for i in range(len(meta['valid_ranges']))]
        self.ranges = np.array(ranges)
        self.eeg_data = np.load(self.path / "sEEG_continuous.npy")
        self.spec_data = np.load(self.path / "Spec_continuous.npy")

        self.eeg_len = int(self.trial_duration * self.fs_eeg)
        self.spec_len = int(self.trial_duration * self.fs_spec)


        if self.mode == 'train':
            np.random.shuffle(self.ranges)  # Shuffle the order of segments for training
        else:
            total_segments = len(self.ranges)
            rng = np.random.RandomState(42)
            val_indices = rng.choice(total_segments, size=total_segments//2, replace=False)
            if self.mode == 'val':
                self.ranges = self.ranges[val_indices]
            elif self.mode == 'test':
                test_mask = np.ones(total_segments, dtype=bool)
                test_mask[val_indices] = False
                self.ranges = self.ranges[test_mask]  

        # filter out the trials that go beyond the data length (should be very few, if any)
        mask = self.ranges[:, -1] <= self.eeg_data.shape[1]/self.fs_eeg   
        self.ranges = self.ranges[mask] 


    def __len__(self):
        return len(self.ranges)
    
    def get_start_win_samples(self, start_time, duration):
        eeg_strt = int(start_time * self.fs_eeg)
        eeg_win = int(duration * self.fs_eeg)
        spect_strt = int(start_time * self.fs_spec)
        spect_win = int(duration * self.fs_spec)
        return (eeg_strt, eeg_win), (spect_strt, spect_win)

    def __getitem__(self, idx):
        start_time, _, _ , end_time = self.ranges[idx]
        eeg_slice, spect_slice = self.get_start_win_samples(start_time, end_time - start_time)

        x = self.eeg_data[:, eeg_slice[0] : eeg_slice[0]+eeg_slice[1]].copy()  # copy to avoid mmap issues
        y = self.spec_data[:, spect_slice[0] : spect_slice[0]+spect_slice[1]].copy()  # copy to avoid mmap issues

        x = (x - np.mean(x, axis=1, keepdims=True)) / (1e-8 + np.std(x, axis=1, keepdims=True)) ## z-scoring

        if self.mode == 'train': ## do augmentations
            rand_strt = np.random.uniform(0, self.max_trial_duration - self.trial_duration)  # random start time
            eeg_sub_slice, spect_sub_slice = self.get_start_win_samples(rand_strt, self.trial_duration)
            x = x[:, eeg_sub_slice[0] : eeg_sub_slice[0] + eeg_sub_slice[1]]
            y = y[:, spect_sub_slice[0] : spect_sub_slice[0] + spect_sub_slice[1]]

            mask = np.random.rand(x.shape[0]) > 0.1
            x = x * mask[:, None]  # broadcast over time
            if random.random() < 0.5:
                x += np.random.normal(0, 0.316, x.shape) ## 0.316 is std dev
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()




@register_dataset("nz_label_predicted")
class SpeechSegmentDataset(Dataset):
    def __init__(self, h5_path, segment_len=172):
        """
        Args:
            h5_path (str): Path to the saved .h5 file.
            segment_len (int): Number of time frames for 2 seconds 
                               (e.g., 160 frames if each frame is 12.5ms).
        """
        self.h5_path = h5_path
        self.segment_len = segment_len
        
        # Open once to get the total number of samples and their full duration
        with h5py.File(self.h5_path, 'r') as f:
            self.num_samples = f['preds'].shape[0]
            self.total_time_frames = f['preds'].shape[2] # Shape: (Batch, 80, Time)
            
        if self.segment_len > self.total_time_frames:
            raise ValueError(f"Segment length ({self.segment_len}) cannot be longer "
                             f"than total frames ({self.total_time_frames})")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # We open the file in the getitem to ensure it's thread-safe for Multi-GPU/Dataloaders
        with h5py.File(self.h5_path, 'r', libver='latest', swmr=True) as f:

            # 2. Extract the slice directly from disk (Efficient!)
            # Shape: [80, segment_len]
            pred_segment = f['preds'][idx, :, :]
            true_segment = f['trues'][idx, :, :]
            
        # Convert to torch tensors
        return torch.from_numpy(pred_segment).float(), torch.from_numpy(true_segment).float()



if __name__ == "__main__":
    dataset = SpeechSegmentDataset("/srv/share/NZ0_spectrograms/train_set.h5", segment_len=172)
    print(f"Total 2-second segments available: {len(dataset)}")
    x, y = dataset[2]
    print(f'random input shape : {x.shape}')
    print(f'random output shape : {y.shape} ')