from torch.utils.data import Dataset
from scipy.stats import zscore
import os, glob, h5py, torch
from tqdm import tqdm 
import numpy as np

# local
from .factory import register_dataset

@register_dataset("nz_unlabel")
class NWBDataset(Dataset):
    def __init__(
        self, root_dir, data_type='Both', duration=2,
        channels_to_keep=None, debug_mode=False, is_train=True,
        trial_duration=None
    ):
        """Dataset class for unlabeled data
        Args:
            root_dir (str): Path to the directory containing two manifest files (train.txt and val.txt)
                train.txt and val.txt should contain the paths to the .nwb files for training and validation respectively.
            data_type (str): HG | LFC | Both | Preprocessed
                HG: High Gamma Envelope (200.0Hz)
                LFC: Preprocessed LFCs (200.0Hz)
                Both: Both High Gamma Envelope and Preprocessed LFCs (200.0Hz)
                Preprocessed: Line noise (and harmonics) filtered + CARed (400.0Hz)
            duration (int): Length of a training sample in seconds
            channels_to_keep: Tell which channels to keep. Must be 0-indexed. Note: EKG channels...
            ... have already been removed internally in this class.
            is_train (bool): Whether to load training data (train.txt) or validation data (val.txt)

        Returns:
            A tuple of (data, sampling_rate) where:
                - data (torch.Tensor): A tensor of shape (num_samples, num_channels)
                - sampling_rate (int): The sampling rate of the data
        """
        self.root_dir = root_dir
        self.duration = duration
        self.data_type = data_type
        self.trial_duration = trial_duration
        self.channels_to_keep = channels_to_keep if channels_to_keep\
            is None else np.array(channels_to_keep)

        self.samp_rate = None
        self.is_train = is_train
        if is_train:
            manifest_path = os.path.join(
                self.root_dir, 'train.txt'
            )
            print(f"Train dataset...")
        else:
            manifest_path = os.path.join(
                self.root_dir, 'val.txt'
            )
            print(f"Validataion dataset...")

        with open(manifest_path, 'r') as m_file:
            nwb_files = [i.strip() for i in m_file.readlines()]

        print(f"Number of files: {len(nwb_files)}")
        if debug_mode:
            nwb_files = nwb_files[:3]

        self.data = []
        
        print('Loading NWB files...')
        
        for ii, nwb in enumerate(tqdm(nwb_files)):
            with h5py.File(
                nwb, mode='r', libver='latest',
                swmr=True
            ) as f:
                if ii == 0:
                    ekg_channels = f['general']['extracellular_ephys']['electrodes']['bad'][:]
                
                if data_type.lower() == 'hg':
                    raw_data = f['processing']['ecephys']['LFP']\
                        ['high gamma (CAR 200.0Hz)']['data'][:, ~ekg_channels]

                    if self.samp_rate is None:
                        self.samp_rate = int(f['processing']['ecephys']['LFP']\
                        ['high gamma (CAR 200.0Hz)']['starting_time'].attrs['rate'].item())

                elif data_type.lower() == 'lfc':
                    raw_data = f['processing']['ecephys']['LFP']\
                        ['preprocessed LFC (CAR 200.0Hz)']['data'][:, ~ekg_channels]

                    if self.samp_rate is None:
                        self.samp_rate = int(f['processing']['ecephys']['LFP']\
                        ['preprocessed LFC (CAR 200.0Hz)']['starting_time'].attrs['rate'].item())
                elif data_type.lower() == 'both':
                    raw_data = np.concatenate(
                        (f['processing']['ecephys']['LFP']\
                            ['high gamma (CAR 200.0Hz)']['data'][:, ~ekg_channels],
                        f['processing']['ecephys']['LFP']\
                            ['preprocessed LFC (CAR 200.0Hz)']['data'][:, ~ekg_channels]),
                        axis=-1
                    )

                    if self.samp_rate is None:
                        hg_rate = int(f['processing']['ecephys']['LFP']\
                        ['high gamma (CAR 200.0Hz)']['starting_time'].attrs['rate'].item())
                        lfc_rate = int(f['processing']['ecephys']['LFP']\
                        ['preprocessed LFC (CAR 200.0Hz)']['starting_time'].attrs['rate'].item())

                        assert hg_rate == lfc_rate, "High Gamma and LFCs must have the same sampling rate."

                        self.samp_rate = hg_rate
                elif data_type.lower() == 'preprocessed':
                    raw_data = f['processing']['ecephys']['LFP']\
                        ['preprocessed (CAR)']['data'][:, ~ekg_channels]

                    if self.samp_rate is None:
                        self.samp_rate = int(f['processing']['ecephys']['LFP']\
                        ['preprocessed (CAR)']['starting_time'].attrs['rate'].item())
                else:
                    raise ValueError("data_type must be 'HG' or 'LFC' or 'Both' or 'Preprocessed'.")
                
                if self.channels_to_keep is None:
                    self.data.append(raw_data)
                else:
                    self.data.append(
                        raw_data[:, self.channels_to_keep]
                    )
        
        print('Successfully loaded NWB files!')

        num_of_duration_segments = [
            len(dat)//(self.duration * self.samp_rate) for dat in self.data
        ]

        self.all_duration_segments = [
            [(nwb_idx, m_idx) for m_idx in range(min_idx)] for nwb_idx, min_idx
            in zip(range(len(self.data)), num_of_duration_segments)
            if min_idx > 0
        ]
        self.all_duration_segments = np.concatenate(self.all_duration_segments).tolist()

    def __len__(self):
        return len(self.all_duration_segments)

    def __getitem__(self, idx):
        nwb_idx, segment_idx = self.all_duration_segments[idx]

        start_idx = segment_idx * self.duration * self.samp_rate
        end_idx = start_idx + self.duration * self.samp_rate
        selected_data = self.data[nwb_idx][start_idx:end_idx]

        norm_data = zscore(
            selected_data,
            axis=0
        )
        if self.is_train and self.trial_duration is not None:
            T = norm_data.shape[0]
            win = int(self.trial_duration*self.samp_rate)
            assert win <= T, f"trial_duration ({win} samples) exceeds trial length ({T} samples)"
            start = np.random.randint(T - win + 1)   # random crop
            norm_data = norm_data[start:start+win]
        tensor_data = torch.tensor(norm_data, dtype=torch.float32)
        return tensor_data, self.samp_rate


if __name__ == "__main__":
    ds = NWBDataset(root_dir="/depot/jgmakin/data/NZ0000/NWB/", data_type='both', duration=2, debug_mode='True') 
    ## debug_mode = True only loads a small number of .nwb files for quick debugging
    print(len(ds))
    random_idx = np.random.randint(len(ds))
    sample_eeg = ds[random_idx]
    print(sample_eeg.shape)
