import shutil
import numpy as np
from pathlib import Path

import torch
import torchvision
import torch.nn as nn 
import torch.nn.functional as F

from datasets import load_dataset, Audio
from abc import ABC, abstractmethod

# local
from diffusion_hub import REPO_ROOT
from .processors import get_processor
from .factory import register_dataset

# logging
import logging
logger = logging.getLogger(__name__)


class BaseDataset(ABC, torch.utils.data.Dataset):
    def __init__(self, data_dir, download=False):
        super().__init__()
        self.data_dir = Path(data_dir)
        if download:
            self._reset_data_dir()
            self._download()

        assert self.data_dir.exists(), f"Dataset not found, make sure to run 'download_data.py' first!"

    def _reset_data_dir(self):
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(parents=True)
        logger.info(f"Initialized clean data directory at {self.data_dir}")

    @abstractmethod
    def _download(self):
        ...




@register_dataset("mnist")
class MNISTDataset:
    """
    General MNIST dataset wrapper.
    Can be instantiated for train or test split.
    Implements __len__ and __getitem__ so it behaves like a standard dataset.
    """
    def __init__(self, data_dir="./data", train=True, download=True):
        self.root = data_dir
        self.train = train
        self.download = download

        # Transform: convert to tensor and scale to [-1, 1]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad(2),                 # pads 2 pixels on all sides: 28+2*2 = 32
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])

        # Initialize underlying torchvision MNIST dataset
        self.dataset = torchvision.datasets.MNIST(
            root=self.root, train=self.train, transform=self.transform, download=self.download
        )

    def __len__(self):
        # Forward to MNIST dataset's len
        return len(self.dataset)

    def __getitem__(self, index):
        # Forward to MNIST dataset's getitem
        return self.dataset[index]

@register_dataset("ljspeech")
class LJSpeechDataset(BaseDataset):
    def __init__(
            self, data_tag, cache_dir, val_metadata,
            max_duration=2, sampling_rate=22050,
            validation=False, download=False,
            processor_config=None,
            ) -> None:
        
        self.data_tag = data_tag
        self.max_duration = max_duration
        self.sampling_rate = sampling_rate
        self.validation = validation
        super(LJSpeechDataset, self).__init__(cache_dir, download=download)

        self.val_metadata = val_metadata
        

        self.data = self._get_data_split(validation)

        # match the sampling rate
        if self.sampling_rate != self.data.features["audio"].sampling_rate:
            self.data = self.data.cast_column('audio', Audio(sampling_rate=self.sampling_rate))
        
        if processor_config is not None:
            self.processor = get_processor(processor_config.name, **processor_config.params)
            self.max_samples = int(max_duration*self.processor.output_sr)
        else:
            self.processor = None
            self.max_samples = int(max_duration*self.sampling_rate)


        
    def _download(self):
        hf_dataset = load_dataset(
            self.data_tag, cache_dir=self.data_dir,
            trust_remote_code=True
            )['train']
        logger.info(f"Dataset downloaded successfully.")

    def _get_data_split(self, validation):
        dataset = load_dataset(
            self.data_tag, cache_dir=self.data_dir,
            trust_remote_code=True
            )['train']
        with open(REPO_ROOT / self.val_metadata, "r") as f:
            evaluation_metadata = f.read().splitlines()
        eval_ids = {line.split("|")[0] for line in evaluation_metadata if line.strip()}
        # ids = dataset["id"]
        # eval_mask = np.array([id_ in eval_ids for id_ in ids])

        val_indices = [i for i, id_ in enumerate(dataset["id"]) if id_ in eval_ids]

        if validation:
            dataset = dataset.select(val_indices)
        else:
            val_set = set(val_indices)
            train_indices = [i for i in range(len(dataset)) if i not in val_set]
            dataset = dataset.select(train_indices)
        return dataset 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]

        audio = example['audio']['array']
        fs = example['audio']['sampling_rate']
        trans = example['normalized_text']

        audio = torch.from_numpy(audio).float().unsqueeze(dim=0)
        if self.processor:
            audio = self.processor(audio)
        return audio, fs, trans
    
    def collate_fn(self, data_list: list):
        """Combines items in the input list and returns a batch"""
        audio_list, fs_list, trans_list = zip(*data_list)
        
        min_length = min(audio.shape[-1] for audio in audio_list)
        max_length = min(min_length, self.max_samples)
        max_length = max_length - (max_length % 8)  # make sure it is divisible by 8
        audio_chunks = []
        for audio in audio_list:
            if audio.shape[-1] > max_length:
                start_idx = np.random.randint(0, audio.shape[-1]-max_length)
            else:
                start_idx = 0
            audio_chunks.append(audio[..., start_idx:start_idx+max_length])

        # clipped_audios = np.stack([audio[..., :max_length] for audio in audio_list])
        clipped_audios = np.stack(audio_chunks)
        clipped_audio_tensors = torch.tensor(clipped_audios, dtype=torch.float32)
        fs_tensor = torch.tensor(np.stack(fs_list))
        return clipped_audio_tensors.squeeze(), fs_tensor
    

class DataCollator:
    def __init__(self, max_duration:int=2, **kwargs) -> None:        
        self.max_duration = max_duration
        self.sampling_rate = 100    # spectrogram sampling rate
        self.max_samples = int(max_duration*self.sampling_rate)
        logger.info(f"DataCollator: max audio duration={self.max_duration}, max samples={self.max_samples}")

    def __call__(self, data_list: list):
        """Combines items in the input list and returns a batch"""
        audio_list, fs_list, trans_list = zip(*data_list)
        
        min_length = min(audio.shape[-1] for audio in audio_list)
        max_length = min(min_length, self.max_samples)
        max_length = max_length - (max_length % 8)  # make sure it is divisible by 8
        audio_chunks = []
        for audio in audio_list:
            if audio.shape[-1] > max_length:
                start_idx = np.random.randint(0, audio.shape[-1]-max_length)
            else:
                start_idx = 0
            audio_chunks.append(audio[..., start_idx:start_idx+max_length])

        # clipped_audios = np.stack([audio[..., :max_length] for audio in audio_list])
        clipped_audios = np.stack(audio_chunks)
        clipped_audio_tensors = torch.tensor(clipped_audios, dtype=torch.float32)
        fs_tensor = torch.tensor(np.stack(fs_list))
        return clipped_audio_tensors.squeeze(), fs_tensor
