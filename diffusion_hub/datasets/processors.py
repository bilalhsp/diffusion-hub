from pathlib import Path

# from librosa.util import normalize
# from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

import torch
from transformers import AutoProcessor

import warnings
import logging
logger = logging.getLogger(__name__)

__PROCESSOR__ = {}


def register_processor(name):
    def wrapper(cls):   
        if __PROCESSOR__.get(name, None):
            if __PROCESSOR__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __PROCESSOR__[name] = cls
        cls.name = name
        return cls
    return wrapper

def get_processor(name: str, **kwargs):
    if __PROCESSOR__.get(name, None) is None:
        raise NameError(f"Dataset '{name}' is not defined.")
    return __PROCESSOR__[name](**kwargs)

def get_supported_processors():
    return list(__PROCESSOR__.keys())



@register_processor("deepspeech2")
class DS2Processor:
    def __init__(self, sample_rate=16000, window_size=0.02, window_stride=0.01, normalize=True) -> None:
        
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.normalize = normalize

        self.output_sr = 1/self.window_stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, wav, **kwds):
        """
        :param y: Audio signal as an array of float numbers
        :return: Spectrogram of the signal
        """
        assert wav.ndim < 3, f"expects 1D or 2D inputs, got {wav.ndim}-dimensional input"
        
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = torch.stft(
            wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=torch.hamming_window(win_length, device=self.device), 
            return_complex=True,
            pad_mode='constant', center=True, normalized=False, onesided=True 
            )
        spect = torch.abs(D)
        spect = torch.log1p(spect)
        if self.normalize:
            spect = (spect - spect.mean()) / (spect.std() + 1e-5)
        return spect.unsqueeze(1)  # add channel dimension

@register_processor("whisper")
class WhisperProcessor:
    def __init__(self, sample_rate=16000, normalize=True) -> None:
        self.sample_rate = sample_rate
        self.normalize = normalize
        repo_name = "openai/whisper-tiny"
        HF_CACHE_DIR = Path('/scratch/gilbreth/ahmedb/cache/hf_cache')
        self.processor = AutoProcessor.from_pretrained(repo_name, cache_dir=HF_CACHE_DIR)

        self.output_sr = self.sample_rate/160
        self.device = torch.device('cpu')

    def __call__(self, wav, **kwds):
        """
        :param wav: Audio signal as an array of float numbers
        :return: Spectrogram of the signal
        """
        assert wav.ndim < 3, f"expects 1D or 2D inputs, got {wav.ndim}-dimensional input"
        
        whisper_mel = self.processor(
            wav.squeeze(), sampling_rate=self.sample_rate, return_tensors="pt"
            ).input_features
        
        whisper_mel = whisper_mel * 4 - 4
        return whisper_mel.unsqueeze(1)  # add channel dimension
    
@register_processor("hifi")
class HifiProcessor:
    def __init__(
            self, sampling_rate=22050, hop_size=256, win_size=1024,
            num_mels=80, n_fft=1024, fmin=0, fmax=8000, center=False,
            use_pow_spec=False,
            ):

        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.win_size = win_size
        self.num_mels = num_mels
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.center=center
        self.use_pow_spec = use_pow_spec

        if self.use_pow_spec:
            logger.info("Using power spectrograms for training.")
        else:
            logger.info("Using magnitude spectrograms (HiFI-GAN) for training.")
        
        self.output_sr = self.sampling_rate/self.hop_size
        self.pad_size = int((self.n_fft-self.hop_size)/2)
        self.device = torch.device('cpu')

        self.mel_basis = {}
        self.hann_window = {}

        if self.fmax not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sampling_rate, n_fft=self.n_fft, 
                n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax
                )
            self.mel_basis[str(fmax)+'_'+str(self.device)] = torch.from_numpy(mel).float().to(self.device)
            self.hann_window[str(self.device)] = torch.hann_window(win_size).to(self.device)

    @staticmethod
    def spectral_normalize_torch(magnitudes):
        output = HifiProcessor.dynamic_range_compression_torch(magnitudes)
        return output
    
    @staticmethod
    def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)
        
    def __call__(self, wav, **kwds):
        """
        :param wav: Audio signal as an array of float numbers
        :return: Spectrogram of the signal
        """
        wav = torch.nn.functional.pad(wav.unsqueeze(1), (self.pad_size, self.pad_size), mode='reflect')
        wav = wav.squeeze(1)

        spec = torch.stft(
            wav, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
            window=self.hann_window[str(self.device)], center=self.center, 
            pad_mode='reflect', normalized=False, onesided=True, return_complex=True
            )
        spec = torch.view_as_real(spec)
        if self.use_pow_spec:
            spec = spec.pow(2).sum(-1)
        else:
            spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
        spec = torch.matmul(self.mel_basis[str(self.fmax)+'_'+str(self.device)], spec)
        spec = HifiProcessor.spectral_normalize_torch(spec)
        return spec

       

def process_input(self, y):
        """
        :param y: Audio signal as an array of float numbers
        :return: Spectrogram of the signal
        """
        assert y.ndim < 3, f"expects 1D or 2D inputs, got {y.ndim}-dimensional input"
        sample_rate = self.config.get('sample_rate', 16000)
        window_size = self.config.get('window_size', 0.02)
        window_stride = self.config.get('window_stride', 0.01)
        window = self.config.get('window', 'hamming')
        normalize = self.config.get('normalize', True)

        n_fft = int(sample_rate * window_size)
        win_length = n_fft
        hop_length = int(sample_rate * window_stride)
        # STFT
        D = torch.stft(
            y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=torch.hamming_window(win_length, device=self.device), 
            return_complex=True,
            pad_mode='constant', center=True, normalized=False, onesided=True 
            )
        spect = torch.abs(D)
        spect = torch.log1p(spect)
        if normalize:
            spect = (spect - spect.mean()) / (spect.std() + 1e-5)
        return spect.unsqueeze(1)  # add channel dimension