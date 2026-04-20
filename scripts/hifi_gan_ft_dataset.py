import os
import numpy as np
from pathlib import Path

from tqdm import tqdm
import torch
from diffusion_hub.datasets import get_processor

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR   = Path("/scratch/gilbreth/ahmedb/data/ljspeech/LJSpeech-1.1")
WAV_DIR    = ROOT_DIR / "wavs"
OUT_DIR    = Path("/scratch/gilbreth/ahmedb/data/ljspeech/spectrograms")

OUT_DIR.mkdir(parents=True, exist_ok=True)

processor = get_processor('hifi', use_pow_spec=True)

# ── Placeholder: replace with your actual function ─────────────────────────
def wav_to_mel(waveform: np.ndarray) -> np.ndarray:
    """
    Args:
        waveform: 1D numpy array
    Returns:
        mel: 2D numpy array (n_mels, T)
    """
    wav_tensor = torch.from_numpy(waveform.squeeze()).float().to(processor.device)
    mel_tensor = processor(wav_tensor[None,:])  # (1, n_mels, T)
    return mel_tensor.squeeze(0).cpu().numpy()  # (n_mels,


# ── Main loop ──────────────────────────────────────────────────────────────
wav_files = sorted(WAV_DIR.glob("*.wav"))
print(f"Found {len(wav_files)} wav files")

iterable = tqdm(enumerate(wav_files), desc="Processing wav files")
for i, wav_path in iterable:
    out_path = OUT_DIR / wav_path.with_suffix(".npy").name

    if out_path.exists():
        print(f"[{i+1}/{len(wav_files)}] Skipping {wav_path.name} (already exists)")
        continue

    try:
        import librosa
        waveform, sr = librosa.load(wav_path, sr=22050)
        mel = wav_to_mel(waveform)                          # (n_mels, T)
        np.save(out_path, mel)
        print(f"[{i+1}/{len(wav_files)}] Saved {out_path.name}  shape={mel.shape}")
    except Exception as e:
        print(f"[{i+1}/{len(wav_files)}] ERROR on {wav_path.name}: {e}")

print(f"\nDone. Spectrograms saved to {OUT_DIR}")