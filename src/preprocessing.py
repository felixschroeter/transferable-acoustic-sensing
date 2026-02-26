"""
preprocessing.py

Contains the code to load the audio files from disk, and apply fourier transform
or short-time fourier transform to it.

"""

import functools

import torch
import torchaudio


@functools.lru_cache(maxsize=10000)
def apply_stft(sound, n_fft=4096, use_log=True, device="cpu"):
    """
    Applies short-time fourier transform to `sound`.
    - `sound`: the input sample
    - `n_fft`: n_fft parameter of the stft
    - `use_log`: if True, apply log to the spectrum
    - `device`: device that the tensor is on, default `cpu`

    generated with ChatGPT and adapted
    """
    sound = sound.to(device)

    # STFT (equivalent to librosa.stft)
    stft = torch.stft(
        sound,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        window=torch.hann_window(n_fft, device=device),
        return_complex=True,
    )

    # Magnitude spectrogram (numpy.abs)
    spectrogram = torch.abs(stft)

    # Sum over time axis (axis=1 in librosa output)
    spectrum = spectrogram.sum(dim=1)

    if use_log:
        spectrum = torch.log(spectrum + 1e-8)
    return spectrum


@functools.lru_cache(maxsize=10000)
def load_audio(path):
    """
    loads the audio stored at path and returns it as a torch tensor
    """
    loaded = torchaudio.load(path)
    x = torch.squeeze(loaded[0])
    return x


@functools.lru_cache(maxsize=None)
def apply_fft(x):
    """
    apply fast fourier transform to `x`
    """
    return torch.fft.rfft(x, norm="forward").real
