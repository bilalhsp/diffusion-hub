import torch
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from pystoi import stoi
from meldataset import mel_spectrogram, get_dataset_filelist
from models import Generator
from env import AttrDict
import json

# ── Config ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "/scratch/gilbreth/ahmedb/data/ljspeech/checkpoints"
WAV_DIR         = Path("/scratch/gilbreth/ahmedb/data/ljspeech/LJSpeech-1.1/wavs")
MEL_DIR         = Path("/scratch/gilbreth/ahmedb/data/ljspeech/spectrograms")
VAL_FILE        = Path("/scratch/gilbreth/ahmedb/data/ljspeech/LJSpeech-1.1/validation.txt")
CONFIG_PATH     = "/scratch/gilbreth/ahmedb/data/ljspeech/checkpoints/config.json"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load config ────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    h = AttrDict(json.load(f))

# ── Load generator from checkpoint ────────────────────────────────────────
def load_generator(checkpoint_path, h, device):
    from utils import scan_checkpoint, load_checkpoint
    cp_g = scan_checkpoint(checkpoint_path, 'g_')
    assert cp_g is not None, "No generator checkpoint found!"
    
    state_dict = load_checkpoint(cp_g, device)
    generator = Generator(h).to(device)
    generator.load_state_dict(state_dict['generator'])
    generator.eval()
    print(f"Loaded checkpoint: {cp_g}")
    return generator

# ── Metrics ────────────────────────────────────────────────────────────────
def compute_mcd(ref_wav, gen_wav, sr=22050, n_mfcc=13):
    ref_mfcc = librosa.feature.mfcc(y=ref_wav, sr=sr, n_mfcc=n_mfcc)
    gen_mfcc = librosa.feature.mfcc(y=gen_wav, sr=sr, n_mfcc=n_mfcc)
    min_len  = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
    diff     = ref_mfcc[:, :min_len] - gen_mfcc[:, :min_len]
    return float((10 / np.log(10)) * np.mean(np.sqrt(2 * np.sum(diff**2, axis=0))))

def compute_stoi(ref_wav, gen_wav, sr=22050):
    min_len = min(len(ref_wav), len(gen_wav))
    return float(stoi(ref_wav[:min_len], gen_wav[:min_len], sr, extended=False))

# ── Evaluation loop ────────────────────────────────────────────────────────
def evaluate(generator, h, device):
    # read validation filelist
    val_files = [line.strip().split('|')[0] for line in open(VAL_FILE)]
    
    mel_errors, mcd_scores, stoi_scores = [], [], []

    with torch.no_grad():
        for i, fname in enumerate(val_files):
            # load ground truth wav
            wav_path = WAV_DIR / f"{fname}.wav"
            mel_path = MEL_DIR / f"{fname}.npy"

            if not wav_path.exists() or not mel_path.exists():
                print(f"Skipping {fname} — file not found")
                continue

            # load mel (your generated spectrogram) and run vocoder
            mel = np.load(mel_path)                                     # (n_mels, T)
            mel = torch.FloatTensor(mel).unsqueeze(0).to(device)        # (1, n_mels, T)

            gen_wav = generator(mel).squeeze().cpu().numpy()            # (T,)

            # load reference wav
            ref_wav, sr = librosa.load(wav_path, sr=h.sampling_rate)

            # mel L1 loss
            mel_ref = mel_spectrogram(
                torch.FloatTensor(ref_wav).unsqueeze(0).to(device),
                h.n_fft, h.num_mels, h.sampling_rate,
                h.hop_size, h.win_size, h.fmin, h.fmax
            )
            mel_gen = mel_spectrogram(
                torch.FloatTensor(gen_wav).unsqueeze(0).to(device),
                h.n_fft, h.num_mels, h.sampling_rate,
                h.hop_size, h.win_size, h.fmin, h.fmax
            )
            mel_errors.append(F.l1_loss(mel_ref, mel_gen).item())

            # MCD and STOI
            mcd_scores.append(compute_mcd(ref_wav, gen_wav, sr=h.sampling_rate))
            stoi_scores.append(compute_stoi(ref_wav, gen_wav, sr=h.sampling_rate))

            print(f"[{i+1}/{len(val_files)}] {fname} | "
                  f"Mel L1: {mel_errors[-1]:.4f} | "
                  f"MCD: {mcd_scores[-1]:.4f} | "
                  f"STOI: {stoi_scores[-1]:.4f}")

    # summary
    print("\n── Results ───────────────────────────────")
    print(f"Mel L1 Error  : {np.mean(mel_errors):.4f} ± {np.std(mel_errors):.4f}")
    print(f"MCD           : {np.mean(mcd_scores):.4f} ± {np.std(mcd_scores):.4f}")
    print(f"STOI          : {np.mean(stoi_scores):.4f} ± {np.std(stoi_scores):.4f}")

    return {
        'mel_l1_mean': float(np.mean(mel_errors)), 'mel_l1_std': float(np.std(mel_errors)),
        'mcd_mean':    float(np.mean(mcd_scores)),  'mcd_std':   float(np.std(mcd_scores)),
        'stoi_mean':   float(np.mean(stoi_scores)), 'stoi_std':  float(np.std(stoi_scores)),
    }

if __name__ == "__main__":
    generator = load_generator(CHECKPOINT_PATH, h, DEVICE)
    results   = evaluate(generator, h, DEVICE)

    # save results
    import json
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved to eval_results.json")