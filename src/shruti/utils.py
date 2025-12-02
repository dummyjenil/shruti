from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import partial
import logging
import math
import os
import random
import librosa
import numpy as np
from omegaconf import DictConfig, OmegaConf
import sentencepiece
import torch
import torch.nn as nn
import torchaudio.functional as F_torchaudio
import torch.nn.functional as F
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
class MelPreprocessor(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_window_size: int = 400,
        n_window_stride: int = 160,
        n_fft: int = 512,
        n_mels: int = 80,
        window="hann",
        preemph: float = 0.97,
        lowfreq: float = 0.0,
        highfreq: float = None,
        power: float = 2.0,         # mag_power
        use_log: bool = True,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: float = 2 ** -24,
        normalize: str = "per_feature",  # or None / "all_features"
        pad_to: int = 16,
        pad_value: float = 0.0,
        mel_norm: str = "slaney"
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.window = window
        self.preemph = preemph
        self.lowfreq = lowfreq
        self.highfreq = highfreq if highfreq is not None else sample_rate / 2.0
        self.power = power
        self.use_log = use_log
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value = log_zero_guard_value
        self.normalize = normalize
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.mel_norm = mel_norm

        # precompute window tensor function (create inside forward so device-aware)
        # Create mel filter matrix once (we will re-create per-device in forward if needed)
        self.register_buffer("_fb_matrix_dummy", torch.tensor([0.0]), persistent=False)

    def _resolve_log_zero_guard_value(self, dtype: torch.dtype) -> float:
        if isinstance(self.log_zero_guard_value, float):
            return self.log_zero_guard_value
        return getattr(torch.finfo(dtype), self.log_zero_guard_value)

    def _preemphasis(self, x: torch.Tensor) -> torch.Tensor:
        if self.preemph is None or self.preemph == 0.0:
            return x
        padded = F.pad(x, (1, 0))
        return x - self.preemph * padded[:, :-1]

    def _compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        # number of frames = floor((length - win_length) / hop_length) + 1
        # But we follow the same formula used in NeMo: length // hop + 1
        return input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()

    def _create_mel_fb(self, device, dtype):
        # Use torchaudio.functional.melscale_fbanks instead of create_fb_matrix
        fb = F_torchaudio.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.lowfreq,
            f_max=self.highfreq,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm=self.mel_norm if self.mel_norm else None,
        )  # shape [n_freqs, n_mels]
        return fb.to(device=device, dtype=dtype)

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor):
        """
        waveform: [B, T] float32
        lengths: [B] int (samples)
        returns: features [B, n_mels, T_frames], feature_lengths [B]
        """
        device = waveform.device
        dtype = waveform.dtype

        # 1) preemphasis
        x = self._preemphasis(waveform)

        # 2) pad center: torchaudio's Spectrogram uses center=True by default; emulate pad
        # Using torch.stft with center=True does internal padding; we'll call it directly.
        # 3) compute stft with return_complex=False -> output shape [B, n_freqs, n_frames, 2]
        # torch.stft args: (input, n_fft, hop_length, win_length, window, center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        window_tensor = getattr(torch, f"{self.window}_window")(self.win_length, dtype=dtype, device=device)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window_tensor,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=False,  # CRITICAL: keep real-valued output (real+imag)
        )  # shape: [B, n_freqs, n_frames, 2]

        # 4) convert to power/magnitude depending on self.power
        real = spec[..., 0]
        imag = spec[..., 1]
        if self.power == 1.0:
            mag = torch.sqrt(real.pow(2) + imag.pow(2)).clamp(min=1e-12)
        elif self.power == 2.0:
            mag = real.pow(2) + imag.pow(2)
        else:
            mag = (real.pow(2) + imag.pow(2)).pow(self.power / 2.0)

        # mag now shape [B, n_freqs, n_frames]
        # 5) apply mel filterbank: mel_FB: [n_freqs, n_mels] -> features = mel_FB^T @ mag
        mel_fb = self._create_mel_fb(device=device, dtype=dtype)  # [n_freqs, n_mels]
        # want [B, n_mels, n_frames] = transpose(mel_fb) @ [B, n_freqs, n_frames]
        # Do einsum for clarity
        feats = torch.einsum("fk,bft->bkt", mel_fb, mag)  # [B, n_mels, n_frames]

        # 6) optional log
        if self.use_log:
            zero_guard = self._resolve_log_zero_guard_value(dtype)
            if self.log_zero_guard_type == "add":
                feats = feats + zero_guard
            elif self.log_zero_guard_type == "clamp":
                feats = feats.clamp(min=zero_guard)
            else:
                raise ValueError("Unsupported log_zero_guard_type")
            feats = feats.log()

        # 7) compute feature lengths
        feat_lens = self._compute_output_lengths(lengths)

        # 8) normalization (per_feature or all_features)
        if self.normalize is not None:
            mask = (torch.arange(feats.shape[-1], device=device).view(1, 1, -1)
                    .repeat(feats.shape[0], 1, 1).lt(feat_lens.view(-1, 1, 1)))
            # mask True where valid -> we want to set pad zeros, so inverse mask for masked_fill
            inv_mask = ~mask
            feats = feats.masked_fill(inv_mask, 0.0)
            # reduce dims
            if self.normalize == "per_feature":
                reduce_dim = 2
            else:
                reduce_dim = [1, 2]
            means = feats.sum(dim=reduce_dim, keepdim=True).div(feat_lens.view(-1, 1, 1).to(feats.dtype))
            stds = (feats - means).masked_fill(inv_mask, 0.0).pow(2.0).sum(dim=reduce_dim, keepdim=True) \
                    .div(feat_lens.view(-1, 1, 1).to(feats.dtype) - 1).clamp(min=self._resolve_log_zero_guard_value(dtype)).sqrt()
            feats = (feats - means) / (stds + CONSTANT)
            feats = feats.masked_fill(inv_mask, 0.0)

        # 9) pad_to if needed
        if self.pad_to > 0 and (feats.shape[-1] % self.pad_to) != 0:
            pad_len = self.pad_to - (feats.shape[-1] % self.pad_to)
            feats = F.pad(feats, (0, pad_len), value=self.pad_value)

        return feats, feat_lens


CONSTANT = 1e-5

def splice_frames(x, frame_splicing):
    """ Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)

def normalize_batch(x, seq_len, normalize_type):
    x_mean = None
    x_std = None
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            # if x[i, :, : seq_len[i]].shape[1] == 1:
            #     raise ValueError(
            #         "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
            #         "in torch.std() returning nan. Make sure your audio length has enough samples for a single "
            #         "feature (ex. at least `hop_length` for Mel Spectrograms)."
            #     )
            x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1), x_mean, x_std
    elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
        x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
        x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)
        return (
            (x - x_mean.view(x.shape[0], x.shape[1]).unsqueeze(2)) / x_std.view(x.shape[0], x.shape[1]).unsqueeze(2),
            x_mean,
            x_std,
        )
    else:
        return x, x_mean, x_std


class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=CONSTANT,
        pad_to=16,
        max_duration=16.7,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()
        if stft_conv or stft_exact_pad:
            logging.warning(
                "Using torch_stft is deprecated and has been removed. The values have been forcibly set to False "
                "for FilterbankFeatures and AudioToMelSpectrogramPreprocessor. Please set exact_pad to True "
                "as needed."
            )
        if exact_pad and n_window_stride % 2 == 1:
            raise NotImplementedError(
                f"{self} received exact_pad == True, but hop_size was odd. If audio_length % hop_size == 0. Then the "
                "returned spectrogram would not be of length audio_length // hop_size. Please use an even hop_size."
            )
        self.log_zero_guard_value = log_zero_guard_value
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        logging.info(f"PADDING: {pad_to}")

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None

        if exact_pad:
            logging.info("STFT using exact pad")
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)
        self.stft = lambda x: torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False if exact_pad else True,
            window=self.window.to(dtype=torch.float),
            return_complex=True,
        )

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq, norm=mel_norm
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob
        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * n_fft)

        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type
        logging.debug(f"sr: {sample_rate}")
        logging.debug(f"n_fft: {self.n_fft}")
        logging.debug(f"win_length: {self.win_length}")
        logging.debug(f"hop_length: {self.hop_length}")
        logging.debug(f"n_mels: {nfilt}")
        logging.debug(f"fmin: {lowfreq}")
        logging.debug(f"fmax: {highfreq}")
        logging.debug(f"using grads: {use_grads}")
        logging.debug(f"nb_augmentation_prob: {nb_augmentation_prob}")

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length) + 1
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    def forward(self, x, seq_len, linear_spec=False):
        seq_len = self.get_seq_len(seq_len)

        if self.stft_pad_amount is not None:
            x = F.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "reflect"
            ).squeeze(1)

        # dither (only in training mode for eval determinism)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)

        # disable autocast to get full range of stft values
        with torch.amp.autocast('cuda',enabled=False):
            x = self.stft(x)

        # torch stft returns complex tensor (of shape [B,N,T]); so convert to magnitude
        # guard is needed for sqrt if grads are passed through
        guard = 0 if not self.use_grads else CONSTANT
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[idx, self._nb_max_fft_bin :, :] = 0.0

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # return plain spectrogram if required
        if linear_spec:
            return x, seq_len

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)
        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
        return x, seq_len


class DummyTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

    # minimum compatibility
    # since all the monolingual tokenizers have a vocab
    # additional methods could be added here
    def get_vocab(self):
        return self.vocab

class TokenizerSpec(ABC):
    """
    Inherit this class to implement a new tokenizer.
    """

    @abstractmethod
    def text_to_tokens(self, text):
        pass

    @abstractmethod
    def tokens_to_text(self, tokens):
        pass

    @abstractmethod
    def tokens_to_ids(self, tokens):
        pass

    @abstractmethod
    def ids_to_tokens(self, ids):
        pass

    @abstractmethod
    def text_to_ids(self, text):
        pass

    @abstractmethod
    def ids_to_text(self, ids):
        pass

    def add_special_tokens(self, special_tokens: List[str]):
        raise NotImplementedError("To be implemented")

    @property
    def name(self):
        return type(self).__name__

    @property
    def unique_identifiers(self):
        """Property required for use with megatron-core datasets."""
        return OrderedDict({"class": f"{type(self).__module__}.{type(self).__qualname__}"})

    @property
    def cls(self):
        """Property alias to match MegatronTokenizer; returns cls_id if available."""
        if hasattr(self, 'cls_id'):
            return self.cls_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'cls' or 'cls_id'")

    @property
    def sep(self):
        """Property alias to match MegatronTokenizer; returns sep_id if available."""
        if hasattr(self, 'sep_id'):
            return self.sep_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'sep' or 'sep_id'")

    @property
    def pad(self):
        """Property alias to match MegatronTokenizer; returns pad_id if available."""
        if hasattr(self, 'pad_id'):
            return self.pad_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'pad' or 'pad_id'")

    @property
    def eod(self):
        """Property alias to match MegatronTokenizer; returns eod_id if available."""
        if hasattr(self, 'eod_id'):
            return self.eod_id
        if hasattr(self, 'eos_id'):
            # Default to end-of-sentence id if end-of-document is not defined.
            return self.eos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'eod', 'eod_id', 'eos', or 'eos_id'")

    @property
    def bos(self):
        """Property alias to match MegatronTokenizer; returns bos_id if available."""
        if hasattr(self, 'bos_id'):
            return self.bos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'bos' or 'bos_id'")

    @property
    def eos(self):
        """Property alias to match MegatronTokenizer; returns eos_id if available."""
        if hasattr(self, 'eos_id'):
            return self.eos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'eos' or 'eos_id'")

    @property
    def mask(self):
        """Property alias to match MegatronTokenizer; returns mask_id if available."""
        if hasattr(self, 'mask_id'):
            return self.mask_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'mask' or 'mask_id'")

class SentencePieceTokenizer(TokenizerSpec):
    """
    Sentencepiecetokenizer https://github.com/google/sentencepiece.
    
    Args:
        model_path: path to sentence piece tokenizer model. To create the model use create_spt_model()
        special_tokens: either list of special tokens or dictionary of token name to token value
        legacy: when set to True, the previous behavior of the SentecePiece wrapper will be restored,
            including the possibility to add special tokens inside wrapper.
    """

    def __init__(
        self, model_path: str, special_tokens: Optional[Union[Dict[str, str], List[str]]] = None, legacy: bool = False
    ):
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"model_path: {model_path} is invalid")
        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.Load(model_path)

        self.original_vocab_size = self.tokenizer.get_piece_size()
        self.vocab_size = self.tokenizer.get_piece_size()
        self.legacy = legacy
        self.special_token_to_id = {}
        self.id_to_special_token = {}
        if special_tokens:
            if not self.legacy:
                raise ValueError(
                    "Special tokens must be None when legacy is set to False. Provide special tokens at train time."
                )
            self.add_special_tokens(special_tokens)
        self.space_sensitive = self.text_to_tokens('x y') != self.text_to_tokens('x') + self.text_to_tokens('y')

    def text_to_tokens(self, text):
        if self.legacy:
            tokens = []
            idx = 0
            last_idx = 0

            while 1:
                indices = {}

                for token in self.special_token_to_id:
                    try:
                        indices[token] = text[idx:].index(token)
                    except ValueError:
                        continue

                if len(indices) == 0:
                    break

                next_token = min(indices, key=indices.get)
                next_idx = idx + indices[next_token]

                tokens.extend(self.tokenizer.encode_as_pieces(text[idx:next_idx]))
                tokens.append(next_token)
                idx = next_idx + len(next_token)

            tokens.extend(self.tokenizer.encode_as_pieces(text[idx:]))
            return tokens

        return self.tokenizer.encode_as_pieces(text)

    def text_to_ids(self, text):
        if self.legacy:
            ids = []
            idx = 0
            last_idx = 0

            while 1:
                indices = {}

                for token in self.special_token_to_id:
                    try:
                        indices[token] = text[idx:].index(token)
                    except ValueError:
                        continue

                if len(indices) == 0:
                    break

                next_token = min(indices, key=indices.get)
                next_idx = idx + indices[next_token]

                ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
                ids.append(self.special_token_to_id[next_token])
                idx = next_idx + len(next_token)

            ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
            return ids

        return self.tokenizer.encode_as_ids(text)

    def tokens_to_text(self, tokens):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        return self.tokenizer.decode_pieces(tokens)

    def ids_to_text(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()

        if self.legacy:
            text = ""
            last_i = 0

            for i, id in enumerate(ids):
                if id in self.id_to_special_token:
                    text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                    text += self.id_to_special_token[id] + " "
                    last_i = i + 1

            text += self.tokenizer.decode_ids(ids[last_i:])
            return text.strip()

        return self.tokenizer.decode_ids(ids)

    def token_to_id(self, token):
        if self.legacy and token in self.special_token_to_id:
            return self.special_token_to_id[token]

        return self.tokenizer.piece_to_id(token)

    def ids_to_tokens(self, ids):
        tokens = []
        for id in ids:
            if id >= self.original_vocab_size:
                tokens.append(self.id_to_special_token[id])
            else:
                tokens.append(self.tokenizer.id_to_piece(id))
        return tokens

    def tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = []
        for token in tokens:
            ids.append(self.token_to_id(token))
        return ids

    def add_special_tokens(self, special_tokens):
        if not self.legacy:
            raise AttributeError("Special Token addition does not work when legacy is set to False.")

        if isinstance(special_tokens, list):
            for token in special_tokens:
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1
        elif isinstance(special_tokens, dict):
            for token_name, token in special_tokens.items():
                setattr(self, token_name, token)
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1

    @property
    def pad_id(self):
        if self.legacy:
            pad_id = self.tokens_to_ids([self.pad_token])[0]
        else:
            pad_id = self.tokenizer.pad_id()
        return pad_id

    @property
    def bos_id(self):
        if self.legacy:
            bos_id = self.tokens_to_ids([self.bos_token])[0]
        else:
            bos_id = self.tokenizer.bos_id()
        return bos_id

    @property
    def eos_id(self):
        if self.legacy:
            eos_id = self.tokens_to_ids([self.eos_token])[0]
        else:
            eos_id = self.tokenizer.eos_id()
        return eos_id

    @property
    def sep_id(self):
        if self.legacy:
            return self.tokens_to_ids([self.sep_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def cls_id(self):
        if self.legacy:
            return self.tokens_to_ids([self.cls_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def mask_id(self):
        if self.legacy:
            return self.tokens_to_ids([self.mask_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def unk_id(self):
        return self.tokenizer.unk_id()

    @property
    def additional_special_tokens_ids(self):
        """Returns a list of the additional special tokens (excluding bos, eos, pad, unk). Used to return sentinel tokens for e.g. T5."""
        special_tokens = set(
            [self.bos_token, self.eos_token, self.pad_token, self.mask_token, self.cls_token, self.sep_token]
        )
        return [v for k, v in self.special_token_to_id.items() if k not in special_tokens]

    @property
    def vocab(self):
        main_vocab = [self.tokenizer.id_to_piece(id) for id in range(self.tokenizer.get_piece_size())]
        special_tokens = [
            self.id_to_special_token[self.original_vocab_size + i]
            for i in range(self.vocab_size - self.original_vocab_size)
        ]
        return main_vocab + special_tokens

class MultilingualTokenizer(TokenizerSpec):
    '''
    MultilingualTokenizer, allowing one to combine multiple regular monolongual tokenizers into one tokenizer.
    The intuition is that we can use existing tokenizers "as is", without retraining, and associate each tokenizer with a language id
    during text processing (language id will be used to route the incoming text sample to the right tokenizer)
    as well as a token id range for detokenization (e.g. [0..127] for tokenizer A, [128..255] for tokenizer B) so
    that the orignal text could be reconstructed. Note that we assume that the incoming dict of langs / tokenizers
    is ordered, e.g. the first tokenizer will be assigned a lower interval of token ids
        Args:
        tokenizers: dict of tokenizers, keys are lang ids, values are actual tokenizers
    '''

    def __init__(self, tokenizers: Dict):

        self.tokenizers_dict = tokenizers
        self.vocabulary = []

        # the tokenizers should produce non-overlapping, ordered token ids
        # keys are language ids
        self.token_id_offset = {}

        # keys are tokenizer numbers
        self.token_id_offset_by_tokenizer_num = {}
        offset = 0
        i = 0
        for lang, tokenizer in self.tokenizers_dict.items():
            self.token_id_offset[lang] = offset
            self.token_id_offset_by_tokenizer_num[i] = offset
            offset += len(tokenizer.vocab)
            i += 1

        for tokenizer in self.tokenizers_dict.values():
            self.vocabulary.extend(tokenizer.vocab)

        self.vocab_size = len(self.vocabulary)
        logging.info(f'Aggregate vocab size: {self.vocab_size}')

        # for compatibility purposes only -- right now only the get_vocab method
        # is supported, returning the joint vocab across all tokenizers
        self.tokenizer = DummyTokenizer(self.vocabulary)

        # lookup tables to speed up token to text operations
        # if there are two tokenizers, [0,1], ['en', 'es'], each with 128 tokens, the aggregate tokenizer
        # token range will be [0,255]. The below method provides three look up tables:
        # one, to convert the incoming token id -- e.g. 200 into its real id (200-127 = 73)
        # second, to compute the tokenizer id that should process that token (1)
        # third, the compute the lang id for that token ('es')
        offset_token_ids_by_token_id, tokenizers_by_token_id, langs_by_token_id = self._calculate_offsets()

        self.offset_token_ids_by_token_id = offset_token_ids_by_token_id
        self.tokenizers_by_token_id = tokenizers_by_token_id
        self.langs_by_token_id = langs_by_token_id

    def _calculate_offsets(self):
        offsets = {}
        tokenizers = {}
        langs = {}
        cur_num = 0
        tot = len(self.tokenizers_dict)
        for id in range(len(self.vocabulary)):
            off_id = id - list(self.token_id_offset.values())[cur_num]
            if cur_num + 1 < tot:
                if id >= list(self.token_id_offset.values())[cur_num + 1]:
                    cur_num += 1
                    off_id = id - list(self.token_id_offset.values())[cur_num]
            offsets[id] = off_id
            tokenizers[id] = list(self.tokenizers_dict.values())[cur_num]
            langs[id] = list(self.tokenizers_dict.keys())[cur_num]

        return offsets, tokenizers, langs

    def text_to_tokens(self, text, lang_id):
        tokenizer = self.tokenizers_dict[lang_id]
        return tokenizer.text_to_tokens(text)

    def text_to_ids(self, text, lang_id):
        tokenizer = self.tokenizers_dict[lang_id]
        token_ids = tokenizer.text_to_ids(text)
        # token_ids[:] = [t + self.token_id_offset[lang_id] for t in token_ids]

        return token_ids

    def tokens_to_text(self, tokens, lang_id):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        tokenizer = self.tokenizers_dict[lang_id]
        return tokenizer.decode_pieces(tokens)

    def ids_to_text(self, ids, lang):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()

        tokens = []
        tokenizer = self.tokenizers_dict[lang]
        for id in ids:
            # offset_id = self.offset_token_ids_by_token_id[id]
            # tokenizer = self.tokenizers_by_token_id[id]
            # tokens.extend(tokenizer.ids_to_tokens([offset_id]))
            tokens.extend(tokenizer.ids_to_tokens([id]))
        text = ''.join(tokens).replace('▁', ' ')

        return text

    def token_to_id(self, token, lang_id):
        tokenizer = self.tokenizers_dict[lang_id]
        return tokenizer.token_to_id(token) + self.token_id_offset[lang_id]

    def ids_to_tokens(self, ids, lang_id):
        tokenizer = self.tokenizers_dict[lang_id]
        tokens = [tokenizer.ids_to_tokens([id])[0] for id in ids]

        return tokens

    def ids_to_text_and_langs(self, ids):
        text_and_langs = []

        for id in ids:
            offset_id = self.offset_token_ids_by_token_id[id]
            tokenizer = self.tokenizers_by_token_id[id]
            token = tokenizer.ids_to_tokens([offset_id])[0]
            text = token.replace('▁', ' ')
            text = text.strip()  # strip for display purposes
            lang = self.langs_by_token_id[id]
            text_and_langs.append({'char': text, 'lang': lang})

        return text_and_langs

    def ids_to_words_and_langs(self, ids):
        words_and_langs = []

        word_ids = []  # tokens belonging to the current word
        for id in ids:
            offset_id = self.offset_token_ids_by_token_id[id]
            tokenizer = self.tokenizers_by_token_id[id]
            token = tokenizer.ids_to_tokens([offset_id])[0]
            if token.startswith('▁'):
                if len(word_ids) > 0:  # if this isn't the first word
                    word = self.ids_to_text(word_ids)
                    word = word.strip()  # strip for display purposes
                    lang = self.ids_to_lang(word_ids)
                    wl = {'word': word, 'lang': lang}
                    words_and_langs.append(wl)
                word_ids = []
            word_ids.append(id)

        if len(word_ids) > 0:  # the last tokens
            word = self.ids_to_text(word_ids)
            word = word.strip()  # strip for display purposes
            lang = self.ids_to_lang(word_ids)
            wl = {'word': word, 'lang': lang}
            words_and_langs.append(wl)

        return words_and_langs

    def ids_to_lang(self, ids):
        lang_cnts = {}

        for id in ids:
            lang = self.langs_by_token_id[id]
            lang_cnt = lang_cnts.get(lang)
            if lang_cnt is not None:
                lang_cnts[lang] = lang_cnt + 1
            else:
                lang_cnts[lang] = 1

        max_lang = ''
        max_lang_cnt = -1
        for lang, lang_cnt in lang_cnts.items():
            if lang_cnt > max_lang_cnt:
                max_lang = lang
                max_lang_cnt = lang_cnt

        return max_lang

    def tokens_to_ids(self, tokens: Union[str, List[str]], langs: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            tokens = [tokens]
        if isinstance(langs, str):
            langs = [langs]

        ids = []
        for i, token in enumerate(tokens):
            lang_id = langs[i]
            ids.append(self.token_to_id(token, lang_id))
        return ids

    @property
    def vocab(self):
        return self.vocabulary

    @property
    def langs(self):
        return list(self.tokenizers_dict.keys())

class LSTMDropout(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: Optional[float],
        forget_gate_bias: Optional[float],
        t_max: Optional[int] = None,
        weights_init_scale: float = 1.0,
        hidden_hidden_bias_scale: float = 0.0,
        proj_size: int = 0,
    ):
        """Returns an LSTM with forget gate bias init to `forget_gate_bias`.
        Args:
            input_size: See `nn.LSTM`.
            hidden_size: See `nn.LSTM`.
            num_layers: See `nn.LSTM`.
            dropout: See `nn.LSTM`.

            forget_gate_bias: float, set by default to 1.0, which constructs a forget gate
                initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

            t_max: int value, set to None by default. If an int is specified, performs Chrono Initialization
                of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
                of training.
                Reference:
                [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)

            weights_init_scale: Float scale of the weights after initialization. Setting to lower than one
                sometimes helps reduce variance between runs.

            hidden_hidden_bias_scale: Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
                the default behaviour.

        Returns:
            A `nn.LSTM`.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, proj_size=proj_size
        )

        if t_max is not None:
            # apply chrono init
            for name, v in self.lstm.named_parameters():
                if 'bias' in name:
                    p = getattr(self.lstm, name)
                    n = p.nelement()
                    hidden_size = n // 4
                    p.data.fill_(0)
                    p.data[hidden_size : 2 * hidden_size] = torch.log(
                        nn.init.uniform_(p.data[0:hidden_size], 1, t_max - 1)
                    )
                    # forget gate biases = log(uniform(1, Tmax-1))
                    p.data[0:hidden_size] = -p.data[hidden_size : 2 * hidden_size]
                    # input gate biases = -(forget gate biases)

        elif forget_gate_bias is not None:
            for name, v in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size] *= float(hidden_hidden_bias_scale)

        self.dropout = nn.Dropout(dropout) if dropout else None

        for name, v in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                v.data *= float(weights_init_scale)

    def forward(
        self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, h = self.lstm(x, h)

        if self.dropout:
            x = self.dropout(x)

        return x, h

class ConfidenceMethodConstants:
    NAMES = ("max_prob", "entropy")
    ENTROPY_TYPES = ("gibbs", "tsallis", "renyi")
    ENTROPY_NORMS = ("lin", "exp")

    @classmethod
    def print(cls):
        return (
            cls.__name__
            + ": "
            + str({"NAMES": cls.NAMES, "ENTROPY_TYPES": cls.ENTROPY_TYPES, "ENTROPY_NORMS": cls.ENTROPY_NORMS})
        )

def get_confidence_measure_bank():
    """Generate a dictionary with confidence measure functionals.

    Supported confidence measures:
        max_prob: normalized maximum probability
        entropy_gibbs_lin: Gibbs entropy with linear normalization
        entropy_gibbs_exp: Gibbs entropy with exponential normalization
        entropy_tsallis_lin: Tsallis entropy with linear normalization
        entropy_tsallis_exp: Tsallis entropy with exponential normalization
        entropy_renyi_lin: Rényi entropy with linear normalization
        entropy_renyi_exp: Rényi entropy with exponential normalization

    Returns:
        dictionary with lambda functions.
    """
    # helper functions
    # Gibbs entropy is implemented without alpha
    neg_entropy_gibbs = lambda x: (x.exp() * x).sum(-1)
    neg_entropy_alpha = lambda x, t: (x * t).exp().sum(-1)
    neg_entropy_alpha_gibbs = lambda x, t: ((x * t).exp() * x).sum(-1)
    # too big for a lambda
    def entropy_tsallis_exp(x, v, t):
        exp_neg_max_ent = math.exp((1 - math.pow(v, 1 - t)) / (1 - t))
        return (((1 - neg_entropy_alpha(x, t)) / (1 - t)).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    def entropy_gibbs_exp(x, v, t):
        exp_neg_max_ent = math.pow(v, -t * math.pow(v, 1 - t))
        return ((neg_entropy_alpha_gibbs(x, t) * t).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    # use Gibbs entropies for Tsallis and Rényi with t == 1.0
    entropy_gibbs_lin_baseline = lambda x, v: 1 + neg_entropy_gibbs(x) / math.log(v)
    entropy_gibbs_exp_baseline = lambda x, v: (neg_entropy_gibbs(x).exp() * v - 1) / (v - 1)
    # fill the measure bank
    confidence_measure_bank = {}
    # Maximum probability measure is implemented without alpha
    confidence_measure_bank["max_prob"] = (
        lambda x, v, t: (x.max(dim=-1)[0].exp() * v - 1) / (v - 1)
        if t == 1.0
        else ((x.max(dim=-1)[0] * t).exp() * math.pow(v, t) - 1) / (math.pow(v, t) - 1)
    )
    confidence_measure_bank["entropy_gibbs_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_alpha_gibbs(x, t) / math.log(v) / math.pow(v, 1 - t)
    )
    confidence_measure_bank["entropy_gibbs_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_gibbs_exp(x, v, t)
    )
    confidence_measure_bank["entropy_tsallis_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + (1 - neg_entropy_alpha(x, t)) / (math.pow(v, 1 - t) - 1)
    )
    confidence_measure_bank["entropy_tsallis_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_tsallis_exp(x, v, t)
    )
    confidence_measure_bank["entropy_renyi_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_alpha(x, t).log2() / (t - 1) / math.log(v, 2)
    )
    confidence_measure_bank["entropy_renyi_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v)
        if t == 1.0
        else (neg_entropy_alpha(x, t).pow(1 / (t - 1)) * v - 1) / (v - 1)
    )
    return confidence_measure_bank

@dataclass
class ConfidenceMethodConfig:
    """A Config which contains the method name and settings to compute per-frame confidence scores.

    Args:
        name: The method name (str).
            Supported values:
                - 'max_prob' for using the maximum token probability as a confidence.
                - 'entropy' for using a normalized entropy of a log-likelihood vector.

        entropy_type: Which type of entropy to use (str).
            Used if confidence_method_cfg.name is set to `entropy`.
            Supported values:
                - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                    the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                    Note that for this entropy, the alpha should comply the following inequality:
                    (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                    where V is the model vocabulary size.
                - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                    Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                    where α is a parameter. When α == 1, it works like the Gibbs entropy.
                    More: https://en.wikipedia.org/wiki/Tsallis_entropy
                - 'renyi' for the Rényi entropy.
                    Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                    where α is a parameter. When α == 1, it works like the Gibbs entropy.
                    More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

        alpha: Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
            When the alpha equals one, scaling is not applied to 'max_prob',
            and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

        entropy_norm: A mapping of the entropy value to the interval [0,1].
            Supported values:
                - 'lin' for using the linear mapping.
                - 'exp' for using exponential mapping with linear shift.
    """

    name: str = "entropy"
    entropy_type: str = "tsallis"
    alpha: float = 0.33
    entropy_norm: str = "exp"
    temperature: str = "DEPRECATED"

    def __post_init__(self):
        if self.temperature != "DEPRECATED":
            # self.temperature has type str
            self.alpha = float(self.temperature)
            self.temperature = "DEPRECATED"
        if self.name not in ConfidenceMethodConstants.NAMES:
            raise ValueError(
                f"`name` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.NAMES) + '`'}. Provided: `{self.name}`"
            )
        if self.entropy_type not in ConfidenceMethodConstants.ENTROPY_TYPES:
            raise ValueError(
                f"`entropy_type` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.ENTROPY_TYPES) + '`'}. Provided: `{self.entropy_type}`"
            )
        if self.alpha <= 0.0:
            raise ValueError(f"`alpha` must be > 0. Provided: {self.alpha}")
        if self.entropy_norm not in ConfidenceMethodConstants.ENTROPY_NORMS:
            raise ValueError(
                f"`entropy_norm` must be one of the following: "
                f"{'`' + '`, `'.join(ConfidenceMethodConstants.ENTROPY_NORMS) + '`'}. Provided: `{self.entropy_norm}`"
            )

class ConfidenceMethodMixin(ABC):
    """Confidence Method Mixin class.

    It initializes per-frame confidence method.
    """

    def _init_confidence_method(self, confidence_method_cfg: Optional[DictConfig] = None):
        """Initialize per-frame confidence method from config.
        """
        # OmegaConf.structured ensures that post_init check is always executed
        confidence_method_cfg = OmegaConf.structured(
            ConfidenceMethodConfig()
            if confidence_method_cfg is None
            else ConfidenceMethodConfig(**confidence_method_cfg)
        )

        # set confidence calculation method
        # we suppose that self.blank_id == len(vocabulary)
        self.num_tokens = (self.blank_id if hasattr(self, "blank_id") else self._blank_index) + 1
        self.alpha = confidence_method_cfg.alpha

        # init confidence measure bank
        self.confidence_measure_bank = get_confidence_measure_bank()

        measure = None
        # construct measure_name
        measure_name = ""
        if confidence_method_cfg.name == "max_prob":
            measure_name = "max_prob"
        elif confidence_method_cfg.name == "entropy":
            measure_name = '_'.join(
                [confidence_method_cfg.name, confidence_method_cfg.entropy_type, confidence_method_cfg.entropy_norm]
            )
        else:
            raise ValueError(f"Unsupported `confidence_method_cfg.name`: `{confidence_method_cfg.name}`")
        if measure_name not in self.confidence_measure_bank:
            raise ValueError(f"Unsupported measure setup: `{measure_name}`")
        measure = partial(self.confidence_measure_bank[measure_name], v=self.num_tokens, t=self.alpha)

        self._confidence_measure = measure

    def _get_confidence(self, x: torch.Tensor) -> list[float]:
        """Compute confidence, return list of confidence items for each item in batch"""
        return self._get_confidence_tensor(x).tolist()

    def _get_confidence_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute confidence, return tensor"""
        return self._confidence_measure(torch.nan_to_num(x))

activation_registry = {
    "identity": nn.Identity,
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}

@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms.

    score: A float score obtained from an AbstractRNNTDecoder module's score_hypothesis method.

    y_sequence: Either a sequence of integer ids pointing to some vocabulary, or a packed torch.Tensor
        behaving in the same manner. dtype must be torch.Long in the latter case.

    dec_state: A list (or list of list) of LSTM-RNN decoder states. Can be None.

    text: (Optional) A decoded string after processing via CTC / RNN-T decoding (removing the CTC/RNNT
        `blank` tokens, and optionally merging word-pieces). Should be used as decoded string for
        Word Error Rate calculation.

    timestep: (Optional) A list of integer indices representing at which index in the decoding
        process did the token appear. Should be of same length as the number of non-blank tokens.

    alignments: (Optional) Represents the CTC / RNNT token alignments as integer tokens along an axis of
        time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of integer indices.
        For RNNT, represented as a dangling list of list of integer indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).
        The set of valid indices **includes** the CTC / RNNT blank token in order to represent alignments.

    frame_confidence: (Optional) Represents the CTC / RNNT per-frame confidence scores as token probabilities
        along an axis of time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of float indices.
        For RNNT, represented as a dangling list of list of float indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).

    token_confidence: (Optional) Represents the CTC / RNNT per-token confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    word_confidence: (Optional) Represents the CTC / RNNT per-word confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    length: Represents the length of the sequence (the original length without padding), otherwise
        defaults to 0.

    y: (Unused) A list of torch.Tensors representing the list of hypotheses.

    lm_state: (Unused) A dictionary state cache used by an external Language Model.

    lm_scores: (Unused) Score of the external Language Model.

    ngram_lm_state: (Optional) State of the external n-gram Language Model.

    tokens: (Optional) A list of decoded tokens (can be characters or word-pieces.

    last_token (Optional): A token or batch of tokens which was predicted in the last step.
    """

    score: float
    y_sequence: Union[List[int], torch.Tensor]
    text: Optional[str] = None
    dec_out: Optional[List[torch.Tensor]] = None
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    timestep: Union[List[int], torch.Tensor] = field(default_factory=list)
    alignments: Optional[Union[List[int], List[List[int]]]] = None
    frame_confidence: Optional[Union[List[float], List[List[float]]]] = None
    token_confidence: Optional[List[float]] = None
    word_confidence: Optional[List[float]] = None
    length: Union[int, torch.Tensor] = 0
    y: List[torch.tensor] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    lm_scores: Optional[torch.Tensor] = None
    ngram_lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    tokens: Optional[Union[List[int], torch.Tensor]] = None
    last_token: Optional[torch.Tensor] = None

    @property
    def non_blank_frame_confidence(self) -> List[float]:
        """Get per-frame confidence for non-blank tokens according to self.timestep

        Returns:
            List with confidence scores. The length of the list is the same as `timestep`.
        """
        non_blank_frame_confidence = []
        # self.timestep can be a dict for RNNT
        timestep = self.timestep['timestep'] if isinstance(self.timestep, dict) else self.timestep
        if len(self.timestep) != 0 and self.frame_confidence is not None:
            if any(isinstance(i, list) for i in self.frame_confidence):  # rnnt
                t_prev = -1
                offset = 0
                for t in timestep:
                    if t != t_prev:
                        t_prev = t
                        offset = 0
                    else:
                        offset += 1
                    non_blank_frame_confidence.append(self.frame_confidence[t][offset])
            else:  # ctc
                non_blank_frame_confidence = [self.frame_confidence[t] for t in timestep]
        return non_blank_frame_confidence

    @property
    def words(self) -> List[str]:
        """Get words from self.text

        Returns:
            List with words (str).
        """
        return [] if self.text is None else self.text.split()

class BatchedHyps:
    """Class to store batched hypotheses (labels, time_indices, scores) for efficient RNNT decoding"""

    def __init__(
        self,
        batch_size: int,
        init_length: int,
        device: Optional[torch.device] = None,
        float_dtype: Optional[torch.dtype] = None,
    ):
        """

        Args:
            batch_size: batch size for hypotheses
            init_length: initial estimate for the length of hypotheses (if the real length is higher, tensors will be reallocated)
            device: device for storing hypotheses
            float_dtype: float type for scores
        """
        if init_length <= 0:
            raise ValueError(f"init_length must be > 0, got {init_length}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self._max_length = init_length

        # batch of current lengths of hypotheses and correspoinding timesteps
        self.current_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        # tensor for storing transcripts
        self.transcript = torch.zeros((batch_size, self._max_length), device=device, dtype=torch.long)
        # tensor for storing timesteps corresponding to transcripts
        self.timesteps = torch.zeros((batch_size, self._max_length), device=device, dtype=torch.long)
        # accumulated scores for hypotheses
        self.scores = torch.zeros(batch_size, device=device, dtype=float_dtype)

        # tracking last timestep of each hyp to avoid infinite looping (when max symbols per frame is restricted)
        # last observed timestep (with label) for each hypothesis
        self.last_timestep = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        # number of labels for the last timestep
        self.last_timestep_lasts = torch.zeros(batch_size, device=device, dtype=torch.long)
        self._batch_indices = torch.arange(batch_size, device=device)
        self._ones_batch = torch.ones_like(self._batch_indices)

    def _allocate_more(self):
        """
        Allocate 2x space for tensors, similar to common C++ std::vector implementations
        to maintain O(1) insertion time complexity
        """
        self.transcript = torch.cat((self.transcript, torch.zeros_like(self.transcript)), dim=-1)
        self.timesteps = torch.cat((self.timesteps, torch.zeros_like(self.timesteps)), dim=-1)
        self._max_length *= 2

    def add_results_(
        self, active_indices: torch.Tensor, labels: torch.Tensor, time_indices: torch.Tensor, scores: torch.Tensor
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses.
        We assume that all tensors have the same first dimension, and labels are non-blanks.
        Args:
            active_indices: tensor with indices of active hypotheses (indices should be within the original batch_size)
            labels: non-blank labels to add
            time_indices: tensor of time index for each label
            scores: label scores
        """
        if active_indices.shape[0] == 0:
            return  # nothing to add
        # if needed - increase storage
        if self.current_lengths.max().item() >= self._max_length:
            self._allocate_more()

        self.add_results_no_checks_(
            active_indices=active_indices, labels=labels, time_indices=time_indices, scores=scores
        )

    def add_results_no_checks_(
        self, active_indices: torch.Tensor, labels: torch.Tensor, time_indices: torch.Tensor, scores: torch.Tensor
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses without checks.
        We assume that all tensors have the same first dimension, and labels are non-blanks.
        Useful if all the memory is pre-allocated, especially with cuda graphs
        (otherwise prefer a more safe `add_results_`)
        Args:
            active_indices: tensor with indices of active hypotheses (indices should be within the original batch_size)
            labels: non-blank labels to add
            time_indices: tensor of time index for each label
            scores: label scores
        """
        # accumulate scores
        self.scores[active_indices] += scores

        # store transcript and timesteps
        active_lengths = self.current_lengths[active_indices]
        self.transcript[active_indices, active_lengths] = labels
        self.timesteps[active_indices, active_lengths] = time_indices
        # store last observed timestep + number of observation for the current timestep
        self.last_timestep_lasts[active_indices] = torch.where(
            self.last_timestep[active_indices] == time_indices, self.last_timestep_lasts[active_indices] + 1, 1
        )
        self.last_timestep[active_indices] = time_indices
        # increase lengths
        self.current_lengths[active_indices] += 1

    def add_results_masked_(
        self, active_mask: torch.Tensor, labels: torch.Tensor, time_indices: torch.Tensor, scores: torch.Tensor
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses.
        We assume that all tensors have the same first dimension, and labels are non-blanks.
        Args:
            active_mask: tensor with mask for active hypotheses (of batch_size)
            labels: non-blank labels to add
            time_indices: tensor of time index for each label
            scores: label scores
        """
        if (self.current_lengths + active_mask).max() >= self._max_length:
            self._allocate_more()
        self.add_results_masked_no_checks_(
            active_mask=active_mask, labels=labels, time_indices=time_indices, scores=scores
        )

    def add_results_masked_no_checks_(
        self, active_mask: torch.Tensor, labels: torch.Tensor, time_indices: torch.Tensor, scores: torch.Tensor
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses without checks.
        We assume that all tensors have the same first dimension, and labels are non-blanks.
        Useful if all the memory is pre-allocated, especially with cuda graphs
        (otherwise prefer a more safe `add_results_`)
        Args:
            active_mask: tensor with mask for active hypotheses (of batch_size)
            labels: non-blank labels to add
            time_indices: tensor of time index for each label
            scores: label scores
        """
        # accumulate scores
        # same as self.scores[active_mask] += scores[active_mask], but non-blocking
        torch.where(active_mask, self.scores + scores, self.scores, out=self.scores)

        # store transcript and timesteps
        self.transcript[self._batch_indices, self.current_lengths] = labels
        self.timesteps[self._batch_indices, self.current_lengths] = time_indices
        # store last observed timestep + number of observation for the current timestep
        # if last_timestep == time_indices, increase; else set to 1
        torch.where(
            torch.logical_and(active_mask, self.last_timestep == time_indices),
            self.last_timestep_lasts + 1,
            self.last_timestep_lasts,
            out=self.last_timestep_lasts,
        )
        torch.where(
            torch.logical_and(active_mask, self.last_timestep != time_indices),
            self._ones_batch,
            self.last_timestep_lasts,
            out=self.last_timestep_lasts,
        )
        # same as: self.last_timestep[active_mask] = time_indices[active_mask], but non-blocking
        torch.where(active_mask, time_indices, self.last_timestep, out=self.last_timestep)
        # increase lengths
        self.current_lengths += active_mask

class BatchedAlignments:
    """
    Class to store batched alignments (logits, labels, frame_confidence).
    Size is different from hypotheses, since blank outputs are preserved
    """

    def __init__(
        self,
        batch_size: int,
        logits_dim: int,
        init_length: int,
        device: Optional[torch.device] = None,
        float_dtype: Optional[torch.dtype] = None,
        store_alignments: bool = True,
        store_frame_confidence: bool = False,
    ):
        """

        Args:
            batch_size: batch size for hypotheses
            logits_dim: dimension for logits
            init_length: initial estimate for the lengths of flatten alignments
            device: device for storing data
            float_dtype: expected logits/confidence data type
            store_alignments: if alignments should be stored
            store_frame_confidence: if frame confidence should be stored
        """
        if init_length <= 0:
            raise ValueError(f"init_length must be > 0, got {init_length}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self.with_frame_confidence = store_frame_confidence
        self.with_alignments = store_alignments
        self._max_length = init_length

        # tensor to store observed timesteps (for alignments / confidence scores)
        self.timesteps = torch.zeros((batch_size, self._max_length), device=device, dtype=torch.long)
        # current lengths of the utterances (alignments)
        self.current_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)

        # empty tensors instead of None to make torch.jit.script happy
        self.logits = torch.zeros(0, device=device, dtype=float_dtype)
        self.labels = torch.zeros(0, device=device, dtype=torch.long)
        if self.with_alignments:
            # logits and labels; labels can contain <blank>, different from BatchedHyps
            self.logits = torch.zeros((batch_size, self._max_length, logits_dim), device=device, dtype=float_dtype)
            self.labels = torch.zeros((batch_size, self._max_length), device=device, dtype=torch.long)

        # empty tensor instead of None to make torch.jit.script happy
        self.frame_confidence = torch.zeros(0, device=device, dtype=float_dtype)
        if self.with_frame_confidence:
            # tensor to store frame confidence
            self.frame_confidence = torch.zeros((batch_size, self._max_length), device=device, dtype=float_dtype)
        self._batch_indices = torch.arange(batch_size, device=device)

    def _allocate_more(self):
        """
        Allocate 2x space for tensors, similar to common C++ std::vector implementations
        to maintain O(1) insertion time complexity
        """
        self.timesteps = torch.cat((self.timesteps, torch.zeros_like(self.timesteps)), dim=-1)
        if self.with_alignments:
            self.logits = torch.cat((self.logits, torch.zeros_like(self.logits)), dim=1)
            self.labels = torch.cat((self.labels, torch.zeros_like(self.labels)), dim=-1)
        if self.with_frame_confidence:
            self.frame_confidence = torch.cat((self.frame_confidence, torch.zeros_like(self.frame_confidence)), dim=-1)
        self._max_length *= 2

    def add_results_(
        self,
        active_indices: torch.Tensor,
        time_indices: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses.
        All tensors must use the same fixed batch dimension.
        Args:
            active_mask: tensor with mask for active hypotheses (of batch_size)
            logits: tensor with raw network outputs
            labels: tensor with decoded labels (can contain blank)
            time_indices: tensor of time index for each label
            confidence: optional tensor with confidence for each item in batch
        """
        # we assume that all tensors have the same first dimension
        if active_indices.shape[0] == 0:
            return  # nothing to add

        # if needed - increase storage
        if self.current_lengths.max().item() >= self._max_length:
            self._allocate_more()

        active_lengths = self.current_lengths[active_indices]
        # store timesteps - same for alignments / confidence
        self.timesteps[active_indices, active_lengths] = time_indices

        if self.with_alignments and logits is not None and labels is not None:
            self.logits[active_indices, active_lengths] = logits
            self.labels[active_indices, active_lengths] = labels

        if self.with_frame_confidence and confidence is not None:
            self.frame_confidence[active_indices, active_lengths] = confidence
        # increase lengths
        self.current_lengths[active_indices] += 1

    def add_results_masked_(
        self,
        active_mask: torch.Tensor,
        time_indices: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses.
        All tensors must use the same fixed batch dimension.
        Args:
            active_mask: tensor with indices of active hypotheses (indices should be within the original batch_size)
            time_indices: tensor of time index for each label
            logits: tensor with raw network outputs
            labels: tensor with decoded labels (can contain blank)
            confidence: optional tensor with confidence for each item in batch
        """
        if (self.current_lengths + active_mask).max() >= self._max_length:
            self._allocate_more()
        self.add_results_masked_no_checks_(
            active_mask=active_mask, time_indices=time_indices, logits=logits, labels=labels, confidence=confidence
        )

    def add_results_masked_no_checks_(
        self,
        active_mask: torch.Tensor,
        time_indices: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ):
        """
        Add results (inplace) from a decoding step to the batched hypotheses.
        All tensors must use the same fixed batch dimension.
        Useful if all the memory is pre-allocated, especially with cuda graphs
        (otherwise prefer a more safe `add_results_masked_`)
        Args:
            active_mask: tensor with indices of active hypotheses (indices should be within the original batch_size)
            time_indices: tensor of time index for each label
            logits: tensor with raw network outputs
            labels: tensor with decoded labels (can contain blank)
            confidence: optional tensor with confidence for each item in batch
        """
        # store timesteps - same for alignments / confidence
        self.timesteps[self._batch_indices, self.current_lengths] = time_indices

        if self.with_alignments and logits is not None and labels is not None:
            self.timesteps[self._batch_indices, self.current_lengths] = time_indices
            self.logits[self._batch_indices, self.current_lengths] = logits
            self.labels[self._batch_indices, self.current_lengths] = labels

        if self.with_frame_confidence and confidence is not None:
            self.frame_confidence[self._batch_indices, self.current_lengths] = confidence
        # increase lengths
        self.current_lengths += active_mask

def avoid_float16_autocast_context():
    """
    If the current autocast context is float16, cast it to bfloat16
    if available (unless we're in jit) or float32
    """

    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return torch.amp.autocast('cuda',dtype=torch.float32)

        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast('cuda',dtype=torch.bfloat16)
        else:
            return torch.amp.autocast('cuda',dtype=torch.float32)
    else:
        return nullcontext()

class Swish(nn.SiLU):
    """
    Swish activation function introduced in 'https://arxiv.org/abs/1710.05941'
    Mathematically identical to SiLU. See note in nn.SiLU for references.
    """

class CausalConv2D(nn.Conv2d):
    """
    A causal version of nn.Conv2d where each location in the 2D matrix would have no access to locations on its right or down
    All arguments are the same as nn.Conv2d except padding which should be set as None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        if padding is not None:
            raise ValueError("Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1

        padding = 0
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(
        self, x,
    ):
        x = F.pad(x, pad=(self._left_padding, self._right_padding, self._left_padding, self._right_padding))
        x = super().forward(x)
        return x

class GreedyBatchedRNNTLoopLabelsComputer(ConfidenceMethodMixin):
    """
    Loop Labels algorithm implementation. Callable.
    """

    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        """
        Init method.
        Args:
            decoder: Prediction network from RNN-T
            joint: Joint module from RNN-T
            blank_index: index of blank symbol
            max_symbols_per_step: max symbols to emit on each step (to avoid infinite looping)
            preserve_alignments: if alignments are needed
            preserve_frame_confidence: if frame confidence is needed
            confidence_method_cfg: config for the confidence
        """
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        self._blank_index = blank_index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
        assert self._SOS == self._blank_index  # "blank as pad" algorithm only

    def __call__(
        self, x: torch.Tensor, out_len: torch.Tensor,
    ) -> Tuple[BatchedHyps, Optional[BatchedAlignments], Any]:
        """
        Optimized batched greedy decoding.
        Iterates over labels, on each step finding the next non-blank label
        (evaluating Joint multiple times in inner loop); It uses a minimal possible amount of calls
        to prediction network (with maximum possible batch size),
        which makes it especially useful for scaling the prediction network.
        During decoding all active hypotheses ("texts") have the same lengths.

        Args:
            x: output from the encoder
            out_len: lengths of the utterances in `x`
        """
        batch_size, max_time, _unused = x.shape
        device = x.device

        x = self.joint.project_encoder(x)  # do not recalculate joint projection, project only once

        # init output structures: BatchedHyps (for results), BatchedAlignments + last decoder state
        # init empty batched hypotheses
        batched_hyps = BatchedHyps(
            batch_size=batch_size,
            init_length=max_time * self.max_symbols if self.max_symbols is not None else max_time,
            device=x.device,
            float_dtype=x.dtype,
        )
        # sample state, will be replaced further when the decoding for hypothesis is done
        last_decoder_state = self.decoder.initialize_state(x)
        # init alignments if necessary
        use_alignments = self.preserve_alignments or self.preserve_frame_confidence
        # always use alignments variable - for torch.jit adaptation, but keep it as minimal as possible
        alignments = BatchedAlignments(
            batch_size=batch_size,
            logits_dim=self.joint.num_classes_with_blank,
            init_length=max_time * 2 if use_alignments else 1,  # blank for each timestep + text tokens
            device=x.device,
            float_dtype=x.dtype,
            store_alignments=self.preserve_alignments,
            store_frame_confidence=self.preserve_frame_confidence,
        )

        # initial state, needed for torch.jit to compile (cannot handle None)
        state = self.decoder.initialize_state(x)
        # indices of elements in batch (constant)
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
        # last found labels - initially <SOS> (<blank>) symbol
        labels = torch.full_like(batch_indices, fill_value=self._SOS)

        # time indices
        time_indices = torch.zeros_like(batch_indices)
        safe_time_indices = torch.zeros_like(time_indices)  # time indices, guaranteed to be < out_len
        time_indices_current_labels = torch.zeros_like(time_indices)
        last_timesteps = out_len - 1

        # masks for utterances in batch
        active_mask: torch.Tensor = out_len > 0
        advance_mask = torch.empty_like(active_mask)

        # for storing the last state we need to know what elements became "inactive" on this step
        active_mask_prev = torch.empty_like(active_mask)
        became_inactive_mask = torch.empty_like(active_mask)

        # loop while there are active utterances
        first_step = True
        while active_mask.any():
            active_mask_prev.copy_(active_mask, non_blocking=True)
            # stage 1: get decoder (prediction network) output
            if first_step:
                # start of the loop, SOS symbol is passed into prediction network, state is None
                # we need to separate this for torch.jit
                decoder_output, state, *_ = self.decoder.predict(
                    labels.unsqueeze(1), None, add_sos=False, batch_size=batch_size
                )
                first_step = False
            else:
                decoder_output, state, *_ = self.decoder.predict(
                    labels.unsqueeze(1), state, add_sos=False, batch_size=batch_size
                )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            # stage 2: get joint output, iteratively seeking for non-blank labels
            # blank label in `labels` tensor means "end of hypothesis" (for this index)
            logits = (
                self.joint.joint_after_projection(x[batch_indices, safe_time_indices].unsqueeze(1), decoder_output,)
                .squeeze(1)
                .squeeze(1)
            )
            scores, labels = logits.max(-1)

            # search for non-blank labels using joint, advancing time indices for blank labels
            # checking max_symbols is not needed, since we already forced advancing time indices for such cases
            blank_mask = labels == self._blank_index
            time_indices_current_labels.copy_(time_indices, non_blocking=True)
            if use_alignments:
                if self.preserve_frame_confidence:
                    logits = F.log_softmax(logits, dim=-1)
                alignments.add_results_masked_(
                    active_mask=active_mask,
                    time_indices=time_indices_current_labels,
                    logits=logits if self.preserve_alignments else None,
                    labels=labels if self.preserve_alignments else None,
                    confidence=self._get_confidence_tensor(logits) if self.preserve_frame_confidence else None,
                )

            # advance_mask is a mask for current batch for searching non-blank labels;
            # each element is True if non-blank symbol is not yet found AND we can increase the time index
            time_indices += blank_mask
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            torch.less(time_indices, out_len, out=active_mask)
            torch.logical_and(active_mask, blank_mask, out=advance_mask)

            # inner loop: find next non-blank labels (if exist)
            while advance_mask.any():
                # same as: time_indices_current_labels[advance_mask] = time_indices[advance_mask], but non-blocking
                # store current time indices to use further for storing the results
                torch.where(advance_mask, time_indices, time_indices_current_labels, out=time_indices_current_labels)
                logits = (
                    self.joint.joint_after_projection(
                        x[batch_indices, safe_time_indices].unsqueeze(1), decoder_output,
                    )
                    .squeeze(1)
                    .squeeze(1)
                )
                # get labels (greedy) and scores from current logits, replace labels/scores with new
                # labels[advance_mask] are blank, and we are looking for non-blank labels
                more_scores, more_labels = logits.max(-1)
                # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
                torch.where(advance_mask, more_labels, labels, out=labels)
                # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
                torch.where(advance_mask, more_scores, scores, out=scores)

                if use_alignments:
                    if self.preserve_frame_confidence:
                        logits = F.log_softmax(logits, dim=-1)
                    alignments.add_results_masked_(
                        active_mask=advance_mask,
                        time_indices=time_indices_current_labels,
                        logits=logits if self.preserve_alignments else None,
                        labels=more_labels if self.preserve_alignments else None,
                        confidence=self._get_confidence_tensor(logits) if self.preserve_frame_confidence else None,
                    )

                blank_mask = labels == self._blank_index
                time_indices += blank_mask
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                torch.less(time_indices, out_len, out=active_mask)
                torch.logical_and(active_mask, blank_mask, out=advance_mask)

            # stage 3: filter labels and state, store hypotheses
            # select states for hyps that became inactive (is it necessary?)
            # this seems to be redundant, but used in the `loop_frames` output
            torch.ne(active_mask, active_mask_prev, out=became_inactive_mask)
            self.decoder.batch_replace_states_mask(
                src_states=state, dst_states=last_decoder_state, mask=became_inactive_mask,
            )

            # store hypotheses
            if self.max_symbols is not None:
                # pre-allocated memory, no need for checks
                batched_hyps.add_results_masked_no_checks_(
                    active_mask, labels, time_indices_current_labels, scores,
                )
            else:
                # auto-adjusted storage
                batched_hyps.add_results_masked_(
                    active_mask, labels, time_indices_current_labels, scores,
                )

            # stage 4: to avoid looping, go to next frame after max_symbols emission
            if self.max_symbols is not None:
                # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
                # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
                force_blank_mask = torch.logical_and(
                    active_mask,
                    torch.logical_and(
                        torch.logical_and(
                            labels != self._blank_index, batched_hyps.last_timestep_lasts >= self.max_symbols,
                        ),
                        batched_hyps.last_timestep == time_indices,
                    ),
                )
                time_indices += force_blank_mask  # emit blank => advance time indices
                # update safe_time_indices, non-blocking
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                # same as: active_mask = time_indices < out_len
                torch.less(time_indices, out_len, out=active_mask)
        if use_alignments:
            return batched_hyps, alignments, last_decoder_state
        return batched_hyps, None, last_decoder_state

class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would have limited access to locations on its right or left
    All arguments are the same as nn.Conv1d except padding.

    If padding is set None, then paddings are set automatically to make it a causal convolution where each location would not see any steps on its right.

    If padding is set as a list (size of 2), then padding[0] would be used as left padding and padding[1] as right padding.
    It would make it possible to control the number of steps to be accessible on the right and left.
    This mode is not supported when stride > 1. padding[0]+padding[1] should be equal to (kernel_size - 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, : -self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1) :]
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        else:
            return x, cache

def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)

class FusedBatchNorm1d(nn.Module):
    """
    Fused BatchNorm to use in Conformer to improve accuracy in finetuning with TTS scenario
    Drop-in replacement for BatchNorm1d with simple affine projection
    """

    def __init__(self, num_features: int):
        """
        Args:
            num_features: number of channels, see original BatchNorm1d documentation
        """
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    @classmethod
    def from_batchnorm(cls, bn: nn.BatchNorm1d):
        """
        Construct FusedBatchNorm1d module from BatchNorm1d
        Args:
            bn: original BatchNorm module

        Returns:
            FusedBatchNorm1d module with initialized params; in eval mode result is equivalent to original BatchNorm
        """
        assert isinstance(bn, nn.BatchNorm1d)
        fused_bn = FusedBatchNorm1d(bn.num_features)
        # init projection params from original batch norm
        # so, for inference mode output is the same
        std = torch.sqrt(bn.running_var.data + bn.eps)
        fused_bn.weight.data = bn.weight.data / std
        fused_bn.bias.data = bn.bias.data - bn.running_mean.data * fused_bn.weight.data
        return fused_bn

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            return x * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1)
        assert x.dim() == 2
        return x * self.weight + self.bias

class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish()):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def reset_parameters_ff(self):
        ffn1_max = self.d_model ** -0.5
        ffn2_max = self.d_ff ** -0.5
        with torch.no_grad():
            nn.init.uniform_(self.linear1.weight, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear1.bias, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear2.weight, -ffn2_max, ffn2_max)
            nn.init.uniform_(self.linear2.bias, -ffn2_max, ffn2_max)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer of Transformer.
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate, max_cache_len=0):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        self.cache_drop_size = None
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._max_cache_len = max_cache_len

    def forward_qkv(self, query, key, value):
        """Transforms query, key and value.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value (torch.Tensor): (batch, time2, size)
        returns:
            q (torch.Tensor): (batch, head, time1, size)
            k (torch.Tensor): (batch, head, time2, size)
            v (torch.Tensor): (batch, head, time2, size)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): (batch, time2, size)
            scores(torch.Tensor): (batch, time1, time2)
            mask(torch.Tensor): (batch, time1, time2)
        returns:
            value (torch.Tensor): transformed `value` (batch, time2, d_model) weighted by the attention scores
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, pos_emb=None, cache=None):
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            cache (torch.Tensor) : (batch, time_cache, size)

        returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
            out = self.forward_attention(v, scores, mask)
        if cache is None:
            return out
        else:
            return out, cache

    def update_cache(self, key, value, query, cache):
        if cache is not None:
            key = value = torch.cat([cache, key], dim=1)
            q_keep_size = query.shape[1] - self.cache_drop_size
            cache = torch.cat([cache[:, q_keep_size:, :], query[:, :q_keep_size, :]], dim=1)
        return key, value, query, cache

class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        pointwise_activation (str): name of the activation function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_` which is treated as the original default from
            the paper.
    """

    def __init__(
        self, d_model, kernel_size, norm_type='batch_norm', conv_context_size=None, pointwise_activation='glu_'
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        if pointwise_activation in activation_registry:
            self.pointwise_activation = activation_registry[pointwise_activation]()
            dw_conv_input_dim = d_model * 2

            if hasattr(self.pointwise_activation, 'inplace'):
                self.pointwise_activation.inplace = True
        else:
            self.pointwise_activation = pointwise_activation
            dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim,
            out_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=dw_conv_input_dim,
            bias=True,
        )

        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)
        elif norm_type == 'instance_norm':
            self.batch_norm = nn.InstanceNorm1d(dw_conv_input_dim)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(dw_conv_input_dim)
        elif norm_type == 'fused_batch_norm':
            self.batch_norm = FusedBatchNorm1d(dw_conv_input_dim)
        elif norm_type.startswith('group_norm'):
            num_groups = int(norm_type.replace("group_norm", ""))
            self.batch_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        # Compute the activation function or use GLU for original Conformer
        if self.pointwise_activation == 'glu_':
            x = nn.functional.glu(x, dim=1)
        else:
            x = self.pointwise_activation(x)

        if pad_mask is not None:
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache

    def reset_parameters_conv(self):
        pw1_max = pw2_max = self.d_model ** -0.5
        dw_max = self.kernel_size ** -0.5

        with torch.no_grad():
            nn.init.uniform_(self.pointwise_conv1.weight, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv1.bias, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv2.weight, -pw2_max, pw2_max)
            nn.init.uniform_(self.pointwise_conv2.bias, -pw2_max, pw2_max)
            nn.init.uniform_(self.depthwise_conv.weight, -dw_max, dw_max)
            nn.init.uniform_(self.depthwise_conv.bias, -dw_max, dw_max)

class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v, max_cache_len=0):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head=n_head, n_feat=n_feat, dropout_rate=dropout_rate, max_cache_len=max_cache_len)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            # nn.init.normal_(self.pos_bias_u, 0.0, 0.02)
            # nn.init.normal_(self.pos_bias_v, 0.0, 0.02)
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = F.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
            cache (torch.Tensor) : (batch, time_cache, size)

        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)

            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)

            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            # (batch, head, time1, time2)
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

            # compute matrix b and matrix d
            # (batch, head, time1, time2)
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)
            # drops extra elements in the matrix_bd to match the matrix_ac's size
            matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

            scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)

            out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        else:
            return out, cache


class ConvSubsampling(nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach introduced in:
    VGGNet Subsampling: Transformer-transducer: end-to-end speech recognition with self-attention (https://arxiv.org/pdf/1910.12977.pdf)
    Striding Subsampling: "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong et al. (https://ieeexplore.ieee.org/document/8462506)
    Args:
        subsampling (str): The subsampling technique from {"vggnet", "striding", "dw-striding"}
        subsampling_factor (int): The subsampling factor which should be a power of 2
        subsampling_conv_chunking_factor (int): Input chunking factor which can be -1 (no chunking) 
        1 (auto) or a power of 2. Default is 1
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(
        self,
        subsampling,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        super().__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        if subsampling == 'vggnet':
            self._stride = 2
            self._kernel_size = 2
            self._ceil_mode = True

            self._left_padding = 0
            self._right_padding = 0

            for i in range(self._sampling_num):
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    nn.Conv2d(
                        in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        ceil_mode=self._ceil_mode,
                    )
                )
                in_channels = conv_channels

        elif subsampling == 'dw_striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            # Layer 1
            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    )
                )
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            groups=in_channels,
                        )
                    )
                else:
                    layers.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        )
                    )

                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'striding_conv1d':

            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv1D(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 1 else conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == 'dw_striding_conv1d':

            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2

            # Layer 1
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        groups=in_channels,
                    ),
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=feat_out if self._sampling_num == 1 else conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    ),
                ]
            )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                layers.extend(
                    [
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        ),
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=feat_out if self._sampling_num == i + 2 else conv_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                        ),
                    ]
                )
                layers.append(activation)
                in_channels = conv_channels

        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        if subsampling in ["vggnet", "dw_striding", "striding"]:

            in_length = torch.tensor(feat_in, dtype=torch.float)
            out_length = calc_length(
                lengths=in_length,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
            self.out = nn.Linear(conv_channels * int(out_length), feat_out)
            self.conv2d_subsampling = True
        elif subsampling in ["striding_conv1d", "dw_striding_conv1d"]:
            self.out = None
            self.conv2d_subsampling = False
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        self.conv = nn.Sequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def forward(self, x:torch.Tensor, lengths:torch.Tensor):
        lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        # Unsqueeze Channel Axis
        if self.conv2d_subsampling:
            x = x.unsqueeze(1)
        # Transpose to Channel First mode
        else:
            x = x.transpose(1, 2)

        # split inputs if chunking_factor is set
        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:
                # if subsampling_conv_chunking_factor is 1, we split only if needed
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                x_ceil = 2 ** 31 / self._conv_channels * self._stride * self._stride
                if torch.numel(x) > x_ceil:
                    need_to_split = True
                else:
                    need_to_split = False
            else:
                # if subsampling_conv_chunking_factor > 1 we always split
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:  # if unable to split by batch, try by channel
                    if self._subsampling == 'dw_striding':
                        x = self.conv_split_by_channel(x)
                    else:
                        x = self.conv(x)  # try anyway
            else:
                x = self.conv(x)
        else:
            x = self.conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)

        return x, lengths

    def reset_parameters(self):
        # initialize weights
        if self._subsampling == 'dw_striding':
            with torch.no_grad():
                # init conv
                scale = 1.0 / self._kernel_size
                dw_max = (self._kernel_size ** 2) ** -0.5
                pw_max = self._conv_channels ** -0.5

                nn.init.uniform_(self.conv[0].weight, -scale, scale)
                nn.init.uniform_(self.conv[0].bias, -scale, scale)

                for idx in range(2, len(self.conv), 3):
                    nn.init.uniform_(self.conv[idx].weight, -dw_max, dw_max)
                    nn.init.uniform_(self.conv[idx].bias, -dw_max, dw_max)
                    nn.init.uniform_(self.conv[idx + 1].weight, -pw_max, pw_max)
                    nn.init.uniform_(self.conv[idx + 1].bias, -pw_max, pw_max)

                # init fc (80 * 64 = 5120 from https://github.com/kssteven418/Squeezeformer/blob/13c97d6cf92f2844d2cb3142b4c5bfa9ad1a8951/src/models/conformer_encoder.py#L487
                fc_scale = (self._feat_out * self._feat_in / self._sampling_num) ** -0.5
                nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
                nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)

    def conv_split_by_batch(self, x:torch.Tensor):
        """ Tries to split input by batch, run conv and concat results """
        b, _, _, _ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
            logging.debug(f'using manually set chunking factor: {cf}')
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2 ** 31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
            cf = 2 ** p
            logging.debug(f'using auto set chunking factor: {cf}')

        new_batch_size = b // cf
        if new_batch_size == 0:  # input is too big
            return x, False

        logging.debug(f'conv subsampling: using split batch size {new_batch_size}')
        return torch.cat([self.conv(chunk) for chunk in torch.split(x, new_batch_size, 0)]), True

    def conv_split_by_channel(self, x):
        """ For dw convs, tries to split input by time, run conv and concat results """
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
                logging.debug(f'using manually set chunking factor: {cf}')
            else:
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                p = math.ceil(math.log(torch.numel(x) / 2 ** 31, 2))
                cf = 2 ** p
                logging.debug(f'using auto set chunking factor: {cf}')

            new_c = int(c // cf)
            if new_c == 0:
                logging.warning(f'chunking factor {cf} is too high; splitting down to one channel.')
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                logging.warning(f'chunking factor {cf} is too high; splitting down to one timestep.')
                new_t = 1

            logging.debug(f'conv dw subsampling: using split C size {new_c} and split T size {new_t}')
            x = self.channel_chunked_conv(self.conv[i * 3 + 2], new_c, x)  # conv2D, depthwise

            # splitting pointwise convs by time
            x = torch.cat([self.conv[i * 3 + 3](chunk) for chunk in torch.split(x, new_t, 2)], 2)  # conv2D, pointwise
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """ Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(
                    chunk, pad=(self._kernel_size - 1, self._stride - 1, self._kernel_size - 1, self._stride - 1)
                )
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)

    def change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int):
        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

class RelPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def __init__(self, d_model, max_len=5000, xscale=None):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale:int = xscale
        self.max_len = max_len

    def create_pe(self, positions):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0)
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length):
        """Reset and extend the positional encodings if needed."""
        needed_size = 2 * length - 1
        if hasattr(self, 'pe') and self.pe.size(1) >= needed_size:
            return
        # positions would be from negative numbers to positive
        # positive positions would be used for left positions and negative for right positions
        self.create_pe(torch.arange(length - 1, -length, -1, dtype=torch.float32).unsqueeze(1))

    def forward(self, x:torch.Tensor, cache_len=0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        return x, pos_emb

def create_masks(padding_length:torch.Tensor, max_audio_length, device):
    att_context_size=[-1, -1]
    att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)
    if att_context_size[0] >= 0:
        att_mask = att_mask.triu(diagonal=-att_context_size[0])
    if att_context_size[1] >= 0:
        att_mask = att_mask.tril(diagonal=att_context_size[1])

    # pad_mask is the masking to be used to ignore paddings
    pad_mask = torch.arange(0, max_audio_length, device=device).expand(
        padding_length.size(0), -1
    ) < padding_length.unsqueeze(-1)

    if att_mask is not None:
        # pad_mask_for_att_mask is the mask which helps to ignore paddings
        pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
        # att_mask is the masking to be used by the MHA layers to ignore the tokens not supposed to be visible
        att_mask = att_mask[:, :max_audio_length, :max_audio_length]
        # paddings should also get ignored, so pad_mask_for_att_mask is used to ignore their corresponding scores
        att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
        att_mask = ~att_mask

    pad_mask = ~pad_mask
    return pad_mask, att_mask

class ConformerLayer(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=[-1, -1],
    ):
        super().__init__()
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # convolution module
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
        )

        # multi-headed self-attention module
        self.norm_self_att = nn.LayerNorm(d_model)
        MHA_max_cache_len = att_context_size[0]

        self.self_attn = RelPositionMultiHeadAttention(
            n_head=n_heads,
            n_feat=d_model,
            dropout_rate=dropout_att,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            max_cache_len=MHA_max_cache_len,
        )

        # second feed forward module
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x:torch.Tensor, att_mask=None, pos_emb=None, pad_mask=None):
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor
        x = self.norm_self_att(residual)
        x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
        residual = residual + self.dropout(x)
        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)
        residual = residual + self.dropout(x)
        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor
        x = self.norm_out(residual)
        # if self.is_access_enabled(getattr(self, "model_guid", None)) and self.access_cfg.get(
        #     'save_encoder_tensors', False
        # ):
        #     self.register_accessible_tensor(name='encoder', tensor=x)
        return x

def label_collate(labels, device=None):
    """Collates the label inputs for the rnn-t prediction network.
    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.
        device: Optional torch device to place the label on.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")

    batch_size = len(labels)
    max_len = max(len(label) for label in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    labels = torch.tensor(cat_labels, dtype=torch.int64, device=device)

    return labels

def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state

def pack_hypotheses(hypotheses: List[Hypothesis], logitlen: torch.Tensor,) -> List[Hypothesis]:

    if hasattr(logitlen, 'cpu'):
        logitlen_cpu = logitlen.to('cpu')
    else:
        logitlen_cpu = logitlen

    for idx, hyp in enumerate(hypotheses):
        hyp.y_sequence = (
            hyp.y_sequence.to(torch.long)
            if isinstance(hyp.y_sequence, torch.Tensor)
            else torch.tensor(hyp.y_sequence, dtype=torch.long)
        )
        hyp.length = logitlen_cpu[idx]

        if hyp.dec_state is not None:
            hyp.dec_state = _states_to_device(hyp.dec_state)

    return hypotheses

class RNNTDecoder(nn.Module):
    """A Recurrent Neural Network Transducer Decoder / Prediction Network (RNN-T Prediction Network).
    An RNN-T Decoder/Prediction network, comprised of a stateful LSTM model.

    Args:
        prednet: A dict-like object which contains the following key-value pairs.

            pred_hidden:
                int specifying the hidden dimension of the prediction net.

            pred_rnn_layers:
                int specifying the number of rnn layers.

            Optionally, it may also contain the following:

                forget_gate_bias:
                    float, set by default to 1.0, which constructs a forget gate
                    initialized to 1.0.
                    Reference:
                    [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

                t_max:
                    int value, set to None by default. If an int is specified, performs Chrono Initialization
                    of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
                    of training.
                    Reference:
                    [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)

                weights_init_scale:
                    Float scale of the weights after initialization. Setting to lower than one
                    sometimes helps reduce variance between runs.

                hidden_hidden_bias_scale:
                    Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
                    the default behaviour.

                dropout:
                    float, set to 0.0 by default. Optional dropout applied at the end of the final LSTM RNN layer.

        vocab_size: int, specifying the vocabulary size of the embedding layer of the Prediction network,
            excluding the RNNT blank token.

        normalization_mode: Can be either None, 'batch' or 'layer'. By default, is set to None.
            Defines the type of normalization applied to the RNN layer.

        random_state_sampling: bool, set to False by default. When set, provides normal-distribution
            sampled state tensors instead of zero tensors during training.
            Reference:
            [Recognizing long-form speech using streaming end-to-end models](https://arxiv.org/abs/1910.11455)

        blank_as_pad: bool, set to True by default. When set, will add a token to the Embedding layer of this
            prediction network, and will treat this token as a pad token. In essence, the RNNT pad token will
            be treated as a pad token, and the embedding layer will return a zero tensor for this token.

            It is set by default as it enables various batch optimizations required for batched beam search.
            Therefore, it is not recommended to disable this flag.
    """

    def __init__(
        self,
        prednet: Dict[str, Any],
        vocab_size: int,
        normalization_mode: Optional[str] = None,
        random_state_sampling: bool = False,
        blank_as_pad: bool = True,
        multisoftmax=False, #CTEMO
        language_masks=None, #CTEMO
        
    ):
        # Required arguments
        self.pred_hidden = prednet['pred_hidden']
        self.pred_rnn_layers = prednet["pred_rnn_layers"]
        self.blank_idx = vocab_size
        self.blank_as_pad = blank_as_pad
        # Initialize the model (blank token increases vocab size by 1)
        super().__init__()

        # Optional arguments
        forget_gate_bias = prednet.get('forget_gate_bias', 1.0)
        t_max = prednet.get('t_max', None)
        weights_init_scale = prednet.get('weights_init_scale', 1.0)
        hidden_hidden_bias_scale = prednet.get('hidden_hidden_bias_scale', 0.0)
        dropout = prednet.get('dropout', 0.0)
        self.random_state_sampling = random_state_sampling
        rnn_hidden_size = prednet.get("rnn_hidden_size", -1)
        
        self.prediction = nn.ModuleDict(
                    {
                        "embed": nn.Embedding(vocab_size + 1, self.pred_hidden, padding_idx=self.blank_idx),
                        "dec_rnn": LSTMDropout(
                            input_size=self.pred_hidden,
                            hidden_size=rnn_hidden_size if rnn_hidden_size > 0 else self.pred_hidden,
                            num_layers=self.pred_rnn_layers,
                            forget_gate_bias=forget_gate_bias,
                            t_max=t_max,
                            dropout=dropout,
                            weights_init_scale=weights_init_scale,
                            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                            proj_size=self.pred_hidden if self.pred_hidden < rnn_hidden_size else 0,),})
        self._rnnt_export = False

        self.multisoftmax = multisoftmax #CTEMO
        self.language_masks = language_masks #CTEMO

    def forward(self, targets, target_length, states=None):
        # y: (B, U)
        y = label_collate(targets)

        # state maintenance is unnecessary during training forward call
        # to get state, use .predict() method.
        if self._rnnt_export:
            add_sos = False
        else:
            add_sos = True

        g, states = self.predict(y, state=states, add_sos=add_sos)  # (B, U, D)
        g = g.transpose(1, 2)  # (B, D, U)

        return g, target_length, states

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Stateful prediction of scores and state for a (possibly null) tokenset.
        This method takes various cases into consideration :
        - No token, no state - used for priming the RNN
        - No token, state provided - used for blank token scoring
        - Given token, states - used for scores + new states

        Here:
        B - batch size
        U - label length
        H - Hidden dimension size of RNN
        L - Number of RNN layers

        Args:
            y: Optional torch tensor of shape [B, U] of dtype long which will be passed to the Embedding.
                If None, creates a zero tensor of shape [B, 1, H] which mimics output of pad-token on EmbeddiNg.

            state: An optional list of states for the RNN. Eg: For LSTM, it is the state list length is 2.
                Each state must be a tensor of shape [L, B, H].
                If None, and during training mode and `random_state_sampling` is set, will sample a
                normal distribution tensor of the above shape. Otherwise, None will be passed to the RNN.

            add_sos: bool flag, whether a zero vector describing a "start of signal" token should be
                prepended to the above "y" tensor. When set, output size is (B, U + 1, H).

            batch_size: An optional int, specifying the batch size of the `y` tensor.
                Can be infered if `y` and `state` is None. But if both are None, then batch_size cannot be None.

        Returns:
            A tuple  (g, hid) such that -

            If add_sos is False:

                g:
                    (B, U, H)

                hid:
                    (h, c) where h is the final sequence hidden state and c is the final cell state:

                        h (tensor), shape (L, B, H)

                        c (tensor), shape (L, B, H)

            If add_sos is True:
                g:
                    (B, U + 1, H)

                hid:
                    (h, c) where h is the final sequence hidden state and c is the final cell state:

                        h (tensor), shape (L, B, H)

                        c (tensor), shape (L, B, H)

        """
        # Get device and dtype of current module
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            if y.device != device:
                y = y.to(device)

            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            # Y is not provided, assume zero tensor with shape [B, 1, H] is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size

            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

        # Prepend blank "start of sequence" symbol (zero tensor)
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        # If in training mode, and random_state_sampling is set,
        # initialize state to random normal distribution tensor.
        if state is None:
            if self.random_state_sampling and self.training:
                state = self.initialize_state(y)

        # Forward step through RNN
        y = y.transpose(0, 1)  # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)  # (B, U + 1, H)

        del y, start, state

        # Adapter module forward step
        # if self.is_adapter_available():
        #     g = self.forward_enabled_adapters(g)

        return g, hid

    def initialize_state(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the state of the LSTM layers, with same dtype and device as input `y`.
        LSTM accepts a tuple of 2 tensors as a state.

        Args:
            y: A torch.Tensor whose device the generated states will be placed on.

        Returns:
            Tuple of 2 tensors, each of shape [L, B, H], where

                L = Number of RNN layers

                B = Batch size

                H = Hidden size of RNN.
        """
        batch = y.size(0)
        if self.random_state_sampling and self.training:
            state = (
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            )

        else:
            state = (
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            )
        return state

    def score_hypothesis(
        self, hypothesis: Hypothesis, cache: Dict[Tuple[int], Any]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Similar to the predict() method, instead this method scores a Hypothesis during beam search.
        Hypothesis is a dataclass representing one hypothesis in a Beam Search.

        Args:
            hypothesis: Refer to Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.

        Returns:
            Returns a tuple (y, states, lm_token) such that:
            y is a torch.Tensor of shape [1, 1, H] representing the score of the last token in the Hypothesis.
            state is a list of RNN states, each of shape [L, 1, H].
            lm_token is the final integer token of the hypothesis.
        """
        if hypothesis.dec_state is not None:
            device = hypothesis.dec_state[0].device
        else:
            _p = next(self.parameters())
            device = _p.device

        # parse "blank" tokens in hypothesis
        if len(hypothesis.y_sequence) > 0 and hypothesis.y_sequence[-1] == self.blank_idx:
            blank_state = True
        else:
            blank_state = False

        # Convert last token of hypothesis to torch.Tensor
        target = torch.full([1, 1], fill_value=hypothesis.y_sequence[-1], device=device, dtype=torch.long)
        lm_token = target[:, -1]  # [1]

        # Convert current hypothesis into a tuple to preserve in cache
        sequence = tuple(hypothesis.y_sequence)

        if sequence in cache:
            y, new_state = cache[sequence]
        else:
            # Obtain score for target token and new states
            if blank_state:
                y, new_state = self.predict(None, state=None, add_sos=False, batch_size=1)  # [1, 1, H]

            else:
                y, new_state = self.predict(
                    target, state=hypothesis.dec_state, add_sos=False, batch_size=1
                )  # [1, 1, H]

            y = y[:, -1:, :]  # Extract just last state : [1, 1, H]
            cache[sequence] = (y, new_state)

        return y, new_state, lm_token

    def batch_score_hypothesis(
        self, hypotheses: List[Hypothesis], cache: Dict[Tuple[int], Any], batch_states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Used for batched beam search algorithms. Similar to score_hypothesis method.

        Args:
            hypothesis: List of Hypotheses. Refer to Hypothesis.
            cache: Dict which contains a cache to avoid duplicate computations.
            batch_states: List of torch.Tensor which represent the states of the RNN for this batch.
                Each state is of shape [L, B, H]

        Returns:
            Returns a tuple (b_y, b_states, lm_tokens) such that:
            b_y is a torch.Tensor of shape [B, 1, H] representing the scores of the last tokens in the Hypotheses.
            b_state is a list of list of RNN states, each of shape [L, B, H].
            Represented as B x List[states].
            lm_token is a list of the final integer tokens of the hypotheses in the batch.
        """
        final_batch = len(hypotheses)

        if final_batch == 0:
            raise ValueError("No hypotheses was provided for the batch!")

        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        tokens = []
        process = []
        done = [None for _ in range(final_batch)]

        # For each hypothesis, cache the last token of the sequence and the current states
        for i, hyp in enumerate(hypotheses):
            sequence = tuple(hyp.y_sequence)

            if sequence in cache:
                done[i] = cache[sequence]
            else:
                tokens.append(hyp.y_sequence[-1])
                process.append((sequence, hyp.dec_state))

        if process:
            batch = len(process)

            # convert list of tokens to torch.Tensor, then reshape.
            tokens = torch.tensor(tokens, device=device, dtype=torch.long).view(batch, -1)
            dec_states = self.initialize_state(tokens.to(dtype=dtype))  # [L, B, H]
            dec_states = self.batch_initialize_states(dec_states, [d_state for seq, d_state in process])

            y, dec_states = self.predict(
                tokens, state=dec_states, add_sos=False, batch_size=batch
            )  # [B, 1, H], List([L, 1, H])

            dec_states = tuple(state.to(dtype=dtype) for state in dec_states)

        # Update done states and cache shared by entire batch.
        j = 0
        for i in range(final_batch):
            if done[i] is None:
                # Select sample's state from the batch state list
                new_state = self.batch_select_state(dec_states, j)

                # Cache [1, H] scores of the current y_j, and its corresponding state
                done[i] = (y[j], new_state)
                cache[process[j][0]] = (y[j], new_state)

                j += 1

        # Set the incoming batch states with the new states obtained from `done`.
        batch_states = self.batch_initialize_states(batch_states, [d_state for y_j, d_state in done])

        # Create batch of all output scores
        # List[1, 1, H] -> [B, 1, H]
        batch_y = torch.stack([y_j for y_j, d_state in done])

        # Extract the last tokens from all hypotheses and convert to a tensor
        lm_tokens = torch.tensor([h.y_sequence[-1] for h in hypotheses], device=device, dtype=torch.long).view(
            final_batch
        )

        return batch_y, batch_states, lm_tokens

    def batch_initialize_states(self, batch_states: List[torch.Tensor], decoder_states: List[List[torch.Tensor]]):
        """
        Create batch of decoder states.

       Args:
           batch_states (list): batch of decoder states
              ([L x (B, H)], [L x (B, H)])

           decoder_states (list of list): list of decoder states
               [B x ([L x (1, H)], [L x (1, H)])]

       Returns:
           batch_states (tuple): batch of decoder states
               ([L x (B, H)], [L x (B, H)])
       """
        # LSTM has 2 states
        new_states = [[] for _ in range(len(decoder_states[0]))]
        for layer in range(self.pred_rnn_layers):
            for state_id in range(len(decoder_states[0])):
                # batch_states[state_id][layer] = torch.stack([s[state_id][layer] for s in decoder_states])
                new_state_for_layer = torch.stack([s[state_id][layer] for s in decoder_states])
                new_states[state_id].append(new_state_for_layer)

        for state_id in range(len(decoder_states[0])):
            new_states[state_id] = torch.stack([state for state in new_states[state_id]])

        return new_states

    def batch_select_state(self, batch_states: List[torch.Tensor], idx: int) -> List[List[torch.Tensor]]:
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (list): batch of decoder states
                ([L x (B, H)], [L x (B, H)])

            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder states for given id
                ([L x (1, H)], [L x (1, H)])
        """
        if batch_states is not None:
            state_list = []
            for state_id in range(len(batch_states)):
                states = [batch_states[state_id][layer][idx] for layer in range(self.pred_rnn_layers)]
                state_list.append(states)

            return state_list
        else:
            return None

    def batch_concat_states(self, batch_states: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Concatenate a batch of decoder state to a packed state.

        Args:
            batch_states (list): batch of decoder states
                B x ([L x (H)], [L x (H)])

        Returns:
            (tuple): decoder states
                (L x B x H, L x B x H)
        """
        state_list = []

        for state_id in range(len(batch_states[0])):
            batch_list = []
            for sample_id in range(len(batch_states)):
                tensor = torch.stack(batch_states[sample_id][state_id])  # [L, H]
                tensor = tensor.unsqueeze(0)  # [1, L, H]
                batch_list.append(tensor)

            state_tensor = torch.cat(batch_list, 0)  # [B, L, H]
            state_tensor = state_tensor.transpose(1, 0)  # [L, B, H]
            state_list.append(state_tensor)

        return state_list

    @classmethod
    def batch_replace_states_mask(
        cls,
        src_states: Tuple[torch.Tensor, torch.Tensor],
        dst_states: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ):
        """Replace states in dst_states with states from src_states using the mask"""
        # same as `dst_states[i][mask] = src_states[i][mask]`, but non-blocking
        # we need to cast, since LSTM is calculated in fp16 even if autocast to bfloat16 is enabled
        dtype = dst_states[0].dtype
        torch.where(mask.unsqueeze(0).unsqueeze(-1), src_states[0].to(dtype), dst_states[0], out=dst_states[0])
        torch.where(mask.unsqueeze(0).unsqueeze(-1), src_states[1].to(dtype), dst_states[1], out=dst_states[1])

    def batch_split_states(
        self, batch_states: Tuple[torch.Tensor, torch.Tensor]
    ) -> list[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Split states into a list of states.
        Useful for splitting the final state for converting results of the decoding algorithm to Hypothesis class.
        """
        return list(zip(batch_states[0].split(1, dim=1), batch_states[1].split(1, dim=1)))

    def batch_copy_states(
        self,
        old_states: List[torch.Tensor],
        new_states: List[torch.Tensor],
        ids: List[int],
        value: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """Copy states from new state to old state at certain indices.

        Args:
            old_states(list): packed decoder states
                (L x B x H, L x B x H)

            new_states: packed decoder states
                (L x B x H, L x B x H)

            ids (list): List of indices to copy states at.

            value (optional float): If a value should be copied instead of a state slice, a float should be provided

        Returns:
            batch of decoder states with partial copy at ids (or a specific value).
                (L x B x H, L x B x H)
        """
        for state_id in range(len(old_states)):
            if value is None:
                old_states[state_id][:, ids, :] = new_states[state_id][:, ids, :]
            else:
                old_states[state_id][:, ids, :] *= 0.0
                old_states[state_id][:, ids, :] += value

        return old_states

    def mask_select_states(
        self, states: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return states by mask selection
        Args:
            states: states for the batch
            mask: boolean mask for selecting states; batch dimension should be the same as for states

        Returns:
            states filtered by mask
        """
        # LSTM in PyTorch returns a tuple of 2 tensors as a state
        return states[0][:, mask], states[1][:, mask]

class RNNTJoint(nn.Module):
    """A Recurrent Neural Network Transducer Joint Network (RNN-T Joint Network).
    An RNN-T Joint network, comprised of a feedforward model.

    Args:
        jointnet: A dict-like object which contains the following key-value pairs.
            encoder_hidden: int specifying the hidden dimension of the encoder net.
            pred_hidden: int specifying the hidden dimension of the prediction net.
            joint_hidden: int specifying the hidden dimension of the joint net
            activation: Activation function used in the joint step. Can be one of
            ['relu', 'tanh', 'sigmoid'].

            Optionally, it may also contain the following:
            dropout: float, set to 0.0 by default. Optional dropout applied at the end of the joint net.

        num_classes: int, specifying the vocabulary size that the joint network must predict,
            excluding the RNNT blank token.

        vocabulary: Optional list of strings/tokens that comprise the vocabulary of the joint network.
            Unused and kept only for easy access for character based encoding RNNT models.

        log_softmax: Optional bool, set to None by default. If set as None, will compute the log_softmax()
            based on the value provided.

        preserve_memory: Optional bool, set to False by default. If the model crashes due to the memory
            intensive joint step, one might try this flag to empty the tensor cache in pytorch.

            Warning: This will make the forward-backward pass much slower than normal.
            It also might not fix the OOM if the GPU simply does not have enough memory to compute the joint.

        fuse_loss_wer: Optional bool, set to False by default.

            Fuses the joint forward, loss forward and
            wer forward steps. In doing so, it trades of speed for memory conservation by creating sub-batches
            of the provided batch of inputs, and performs Joint forward, loss forward and wer forward (optional),
            all on sub-batches, then collates results to be exactly equal to results from the entire batch.

            When this flag is set, prior to calling forward, the fields `loss` and `wer` (either one) *must*
            be set using the `RNNTJoint.set_loss()` or `RNNTJoint.set_wer()` methods.

            Further, when this flag is set, the following argument `fused_batch_size` *must* be provided
            as a non negative integer. This value refers to the size of the sub-batch.

            When the flag is set, the input and output signature of `forward()` of this method changes.
            Input - in addition to `encoder_outputs` (mandatory argument), the following arguments can be provided.

                - decoder_outputs (optional). Required if loss computation is required.

                - encoder_lengths (required)

                - transcripts (optional). Required for wer calculation.

                - transcript_lengths (optional). Required for wer calculation.

                - compute_wer (bool, default false). Whether to compute WER or not for the fused batch.

            Output - instead of the usual `joint` log prob tensor, the following results can be returned.

                - loss (optional). Returned if decoder_outputs, transcripts and transript_lengths are not None.

                - wer_numerator + wer_denominator (optional). Returned if transcripts, transcripts_lengths are provided
                    and compute_wer is set.

        fused_batch_size: Optional int, required if `fuse_loss_wer` flag is set. Determines the size of the
            sub-batches. Should be any value below the actual batch size per GPU.
    """

    def __init__(
        self,
        jointnet: Dict[str, Any],
        num_classes: int,
        num_extra_outputs: int = 0,
        vocabulary: Optional[List] = None,
        log_softmax: Optional[bool] = None,
        preserve_memory: bool = False,
        fuse_loss_wer: bool = False,
        fused_batch_size: Optional[int] = None,
        experimental_fuse_loss_wer: Any = None,
        language_masks=None, #CTEMO
        multilingual: bool = False, #CTEMO
        language_keys: Optional[List] = None, #CTEMO
        token_id_offsets=None, #CTEMO
        offset_token_ids_by_token_id=None, #CTEMO
    ):
        super().__init__()

        self.vocabulary = vocabulary

        self._vocab_size = num_classes
        self._num_extra_outputs = num_extra_outputs
        self._num_classes = num_classes + 1 + num_extra_outputs  # 1 is for blank
        self.language_masks = language_masks #CTEMO
        self.token_id_offsets = token_id_offsets #CTEMO
        self.offset_token_ids_by_token_id = offset_token_ids_by_token_id #CTEMO
        self.multilingual = multilingual #CTEMO
        self.language_keys = language_keys #CTEMO

        if experimental_fuse_loss_wer is not None:
            # Override fuse_loss_wer from deprecated argument
            fuse_loss_wer = experimental_fuse_loss_wer

        self._fuse_loss_wer = fuse_loss_wer
        self._fused_batch_size = fused_batch_size

        if fuse_loss_wer and (fused_batch_size is None):
            raise ValueError("If `fuse_loss_wer` is set, then `fused_batch_size` cannot be None!")

        self._loss = None
        self._wer = None

        # Log softmax should be applied explicitly only for CPU
        self.log_softmax = log_softmax
        self.preserve_memory = preserve_memory

        if preserve_memory:
            logging.warning(
                "`preserve_memory` was set for the Joint Model. Please be aware this will severely impact "
                "the forward-backward step time. It also might not solve OOM issues if the GPU simply "
                "does not have enough memory to compute the joint."
            )

        # Required arguments
        self.encoder_hidden = jointnet['encoder_hidden']
        self.pred_hidden = jointnet['pred_hidden']
        self.joint_hidden = jointnet['joint_hidden']
        self.activation = jointnet['activation']

        # Optional arguments
        dropout = jointnet.get('dropout', 0.0)

        self.pred, self.enc, self.joint_net = self._joint_net_modules(
            num_classes=self._num_classes,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            enc_n_hidden=self.encoder_hidden,
            joint_n_hidden=self.joint_hidden,
            activation=self.activation,
            dropout=dropout,
        )

        # Flag needed for RNNT export support
        self._rnnt_export = False

        # to change, requires running ``model.temperature = T`` explicitly
        self.temperature = 1.0

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_outputs: Optional[torch.Tensor],
        encoder_lengths: Optional[torch.Tensor] = None,
        transcripts: Optional[torch.Tensor] = None,
        transcript_lengths: Optional[torch.Tensor] = None,
        compute_wer: bool = False,
        language_ids=None, #CTEMO
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # encoder = (B, D, T)
        # decoder = (B, D, U) if passed, else None
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)

        if decoder_outputs is not None:
            decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        if not self._fuse_loss_wer:
            if decoder_outputs is None:
                raise ValueError(
                    "decoder_outputs passed is None, and `fuse_loss_wer` is not set. "
                    "decoder_outputs can only be None for fused step!"
                )

            out = self.joint(encoder_outputs, decoder_outputs, language_ids=language_ids)  # [B, T, U, V + 1] #CTEMO
            return out

        else:
            # At least the loss module must be supplied during fused joint
            if self._loss is None or self._wer is None:
                raise ValueError("`fuse_loss_wer` flag is set, but `loss` and `wer` modules were not provided! ")

            # If fused joint step is required, fused batch size is required as well
            if self._fused_batch_size is None:
                raise ValueError("If `fuse_loss_wer` is set, then `fused_batch_size` cannot be None!")

            # When using fused joint step, both encoder and transcript lengths must be provided
            if (encoder_lengths is None) or (transcript_lengths is None):
                raise ValueError(
                    "`fuse_loss_wer` is set, therefore encoder and target lengths " "must be provided as well!"
                )

            losses = []
            wers, wer_nums, wer_denoms = [], [], []
            target_lengths = []
            batch_size = int(encoder_outputs.size(0))  # actual batch size

            # Iterate over batch using fused_batch_size steps
            for batch_idx in range(0, batch_size, self._fused_batch_size):
                begin = batch_idx
                end = min(begin + self._fused_batch_size, batch_size)

                # Extract the sub batch inputs
                # sub_enc = encoder_outputs[begin:end, ...]
                # sub_transcripts = transcripts[begin:end, ...]
                sub_enc = encoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))
                sub_transcripts = transcripts.narrow(dim=0, start=begin, length=int(end - begin))

                sub_enc_lens = encoder_lengths[begin:end]
                sub_transcript_lens = transcript_lengths[begin:end]

                # Sub transcripts does not need the full padding of the entire batch
                # Therefore reduce the decoder time steps to match
                max_sub_enc_length = sub_enc_lens.max()
                max_sub_transcript_length = sub_transcript_lens.max()

                if decoder_outputs is not None:
                    # Reduce encoder length to preserve computation
                    # Encoder: [sub-batch, T, D] -> [sub-batch, T', D]; T' < T
                    if sub_enc.shape[1] != max_sub_enc_length:
                        sub_enc = sub_enc.narrow(dim=1, start=0, length=int(max_sub_enc_length))

                    # sub_dec = decoder_outputs[begin:end, ...]  # [sub-batch, U, D]
                    sub_dec = decoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))  # [sub-batch, U, D]

                    # Reduce decoder length to preserve computation
                    # Decoder: [sub-batch, U, D] -> [sub-batch, U', D]; U' < U
                    if sub_dec.shape[1] != max_sub_transcript_length + 1:
                        sub_dec = sub_dec.narrow(dim=1, start=0, length=int(max_sub_transcript_length + 1))

                    # Perform joint => [sub-batch, T', U', V + 1]
                    if language_ids is not None: #CTEMO
                        sub_joint = self.joint(sub_enc, sub_dec, language_ids=language_ids[begin:end]) #CTEMO
                    else:
                        sub_joint = self.joint(sub_enc, sub_dec) #CTEMO

                    del sub_dec

                    # Reduce transcript length to correct alignment
                    # Transcript: [sub-batch, L] -> [sub-batch, L']; L' <= L
                    if sub_transcripts.shape[1] != max_sub_transcript_length:
                        sub_transcripts = sub_transcripts.narrow(dim=1, start=0, length=int(max_sub_transcript_length))

                    # Compute sub batch loss
                    # preserve loss reduction type
                    loss_reduction = self.loss.reduction

                    # override loss reduction to sum
                    self.loss.reduction = None

                    # compute and preserve loss
                    loss_batch = self.loss(
                        log_probs=sub_joint,
                        targets=sub_transcripts,
                        input_lengths=sub_enc_lens,
                        target_lengths=sub_transcript_lens,
                    )
                    losses.append(loss_batch)
                    target_lengths.append(sub_transcript_lens)

                    # reset loss reduction type
                    self.loss.reduction = loss_reduction

                else:
                    losses = None

                # Update WER for sub batch
                if compute_wer:
                    sub_enc = sub_enc.transpose(1, 2)  # [B, T, D] -> [B, D, T]
                    sub_enc = sub_enc.detach()
                    sub_transcripts = sub_transcripts.detach()

                    # Update WER on each process without syncing
                    if language_ids is not None: #CTEMO
                        self.wer.update(
                            predictions=sub_enc,
                            predictions_lengths=sub_enc_lens,
                            targets=sub_transcripts,
                            targets_lengths=sub_transcript_lens,
                            lang_ids=language_ids[begin:end]
                        )
                    else:
                        self.wer.update(
                            predictions=sub_enc,
                            predictions_lengths=sub_enc_lens,
                            targets=sub_transcripts,
                            targets_lengths=sub_transcript_lens,
                        )
                    # Sync and all_reduce on all processes, compute global WER
                    wer, wer_num, wer_denom = self.wer.compute()
                    self.wer.reset()

                    wers.append(wer)
                    wer_nums.append(wer_num)
                    wer_denoms.append(wer_denom)

                del sub_enc, sub_transcripts, sub_enc_lens, sub_transcript_lens

            # Reduce over sub batches
            if losses is not None:
                losses = self.loss.reduce(losses, target_lengths)

            # Collect sub batch wer results
            if compute_wer:
                wer = sum(wers) / len(wers)
                wer_num = sum(wer_nums)
                wer_denom = sum(wer_denoms)
            else:
                wer = None
                wer_num = None
                wer_denom = None

            return losses, wer, wer_num, wer_denom

    def joint(self, f: torch.Tensor, g: torch.Tensor, language_ids=None) -> torch.Tensor:
        return self.joint_after_projection(self.project_encoder(f), self.project_prednet(g), language_ids)

    def project_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Project the encoder output to the joint hidden dimension.

        Args:
            encoder_output: A torch.Tensor of shape [B, T, D]

        Returns:
            A torch.Tensor of shape [B, T, H]
        """
        return self.enc(encoder_output)

    def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
        """
        Project the Prediction Network (Decoder) output to the joint hidden dimension.

        Args:
            prednet_output: A torch.Tensor of shape [B, U, D]

        Returns:
            A torch.Tensor of shape [B, U, H]
        """
        return self.pred(prednet_output)

    def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor, language_ids=None) -> torch.Tensor: #CTEMO
        """
        Compute the joint step of the network after projection.

        Here,
        B = Batch size
        T = Acoustic model timesteps
        U = Target sequence length
        H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
        H = Hidden dimension of the Joint hidden step.
        V = Vocabulary size of the Decoder (excluding the RNNT blank token).

        NOTE:
            The implementation of this model is slightly modified from the original paper.
            The original paper proposes the following steps :
            (enc, dec) -> Expand + Concat + Sum [B, T, U, H1+H2] -> Forward through joint hidden [B, T, U, H] -- *1
            *1 -> Forward through joint final [B, T, U, V + 1].

            We instead split the joint hidden into joint_hidden_enc and joint_hidden_dec and act as follows:
            enc -> Forward through joint_hidden_enc -> Expand [B, T, 1, H] -- *1
            dec -> Forward through joint_hidden_dec -> Expand [B, 1, U, H] -- *2
            (*1, *2) -> Sum [B, T, U, H] -> Forward through joint final [B, T, U, V + 1].

        Args:
            f: Output of the Encoder model. A torch.Tensor of shape [B, T, H1]
            g: Output of the Decoder model. A torch.Tensor of shape [B, U, H2]

        Returns:
            Logits / log softmaxed tensor of shape (B, T, U, V + 1).
        """
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)
        g = g.unsqueeze(dim=1)  # (B, 1, U, H)
        inp = f + g  # [B, T, U, H]

        del f, g

        # Forward adapter modules on joint hidden
        # if self.is_adapter_available():
        #     inp = self.forward_enabled_adapters(inp)

        if language_ids is not None: #CTEMO
            # Do partial forward of joint net (skipping the final linear)
            for module in self.joint_net[:-1]:
                inp = module(inp)  # [B, T, U, H]
            
            # check if all the items in the batch have the same langauge, pass them through
            if len(set(language_ids)) == 1:
                res = self.joint_net[-1][language_ids[0]](inp)
            else:
                res_single = []
                for single_inp, lang in zip(inp, language_ids):
                    res_single.append(self.joint_net[-1][lang](single_inp))
                res = torch.stack(res_single)
        else:
            res = self.joint_net(inp)  # [B, T, U, V + 1]

        del inp

        if self.preserve_memory:
            torch.cuda.empty_cache()

        # If log_softmax is automatic
        if self.log_softmax is None:
            if not res.is_cuda:  # Use log softmax only if on CPU
                if self.temperature != 1.0:
                    res = (res / self.temperature).log_softmax(dim=-1)
                else:
                    res = res.log_softmax(dim=-1)
        else:
            if self.log_softmax:
                if self.temperature != 1.0:
                    res = (res / self.temperature).log_softmax(dim=-1)
                else:
                    res = res.log_softmax(dim=-1)

        return res

    def _joint_net_modules(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout):
        """
        Prepare the trainable modules of the Joint Network

        Args:
            num_classes: Number of output classes (vocab size) excluding the RNNT blank token.
            pred_n_hidden: Hidden size of the prediction network.
            enc_n_hidden: Hidden size of the encoder network.
            joint_n_hidden: Hidden size of the joint network.
            activation: Activation of the joint. Can be one of [relu, tanh, sigmoid]
            dropout: Dropout value to apply to joint.
        """
        pred = nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = nn.Linear(enc_n_hidden, joint_n_hidden)

        if activation not in ['relu', 'sigmoid', 'tanh']:
            raise ValueError("Unsupported activation for joint step - please pass one of " "[relu, sigmoid, tanh]")

        activation = activation.lower()

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'tanh':
            activation = nn.Tanh()

        if self.multilingual: #CTEMO
            final_layer = nn.ModuleDict()
            logging.info(f"Vocab size for each language: {self._vocab_size // len(self.language_keys)}")
            for lang in self.language_keys:
                final_layer[lang] = nn.Linear(joint_n_hidden, (self._vocab_size // len(self.language_keys)+1))
            layers = (
                [activation]
                + ([nn.Dropout(p=dropout)] if dropout else [])
                + [final_layer]
            )
        else:
            layers = (
                [activation]
                + ([nn.Dropout(p=dropout)] if dropout else [])
                + [nn.Linear(joint_n_hidden, num_classes)]
            )
        return pred, enc, nn.Sequential(*layers)

    @property
    def num_classes_with_blank(self):
        return self._num_classes

    @property
    def num_extra_outputs(self):
        return self._num_extra_outputs

    @property
    def loss(self):
        return self._loss

    def set_loss(self, loss):
        if not self._fuse_loss_wer:
            raise ValueError("Attempting to set loss module even though `fuse_loss_wer` is not set!")

        self._loss = loss

    @property
    def wer(self):
        return self._wer

    def set_wer(self, wer):
        if not self._fuse_loss_wer:
            raise ValueError("Attempting to set WER module even though `fuse_loss_wer` is not set!")

        self._wer = wer

    @property
    def fuse_loss_wer(self):
        return self._fuse_loss_wer

    def set_fuse_loss_wer(self, fuse_loss_wer, loss=None, metric=None):
        self._fuse_loss_wer = fuse_loss_wer

        self._loss = loss
        self._wer = metric

    @property
    def fused_batch_size(self):
        return self._fused_batch_size

    def set_fused_batch_size(self, fused_batch_size):
        self._fused_batch_size = fused_batch_size

class _GreedyRNNTInfer(ConfidenceMethodMixin):
    """A greedy transducer decoder.

    Provides a common abstraction for sample level and batch level greedy decoding.

    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        preserve_alignments: Bool flag which preserves the history of alignments generated during
            greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of
            Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.
        preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores generated
            during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of List of floats.

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more confidence scores.
            U is the number of target tokens for the current timestep Ti.
        confidence_method_cfg: A dict-like object which contains the method name and settings to compute per-frame
            confidence scores.

            name: The method name (str).
                Supported values:
                    - 'max_prob' for using the maximum token probability as a confidence.
                    - 'entropy' for using a normalized entropy of a log-likelihood vector.

            entropy_type: Which type of entropy to use (str). Used if confidence_method_cfg.name is set to `entropy`.
                Supported values:
                    - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                        the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                        Note that for this entropy, the alpha should comply the following inequality:
                        (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                        where V is the model vocabulary size.
                    - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                        Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                        where α is a parameter. When α == 1, it works like the Gibbs entropy.
                        More: https://en.wikipedia.org/wiki/Tsallis_entropy
                    - 'renyi' for the Rényi entropy.
                        Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                        where α is a parameter. When α == 1, it works like the Gibbs entropy.
                        More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

            alpha: Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                When the alpha equals one, scaling is not applied to 'max_prob',
                and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

            entropy_norm: A mapping of the entropy value to the interval [0,1].
                Supported values:
                    - 'lin' for using the linear mapping.
                    - 'exp' for using exponential mapping with linear shift.
    """

    def __init__(
        self,
        decoder_model: RNNTDecoder,
        joint_model: RNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
        preserve_frame_confidence: bool = False,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.decoder = decoder_model
        self.joint = joint_model

        self._blank_index = blank_index
        self._SOS = blank_index  # Start of single index

        if max_symbols_per_step is not None and max_symbols_per_step <= 0:
            raise ValueError(f"Expected max_symbols_per_step > 0 (or None), got {max_symbols_per_step}")
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence

        # set confidence calculation method
        self._init_confidence_method(confidence_method_cfg)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        hidden: Optional[torch.Tensor],
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Common prediction step based on the AbstractRNNTDecoder implementation.

        Args:
            label: (int/torch.Tensor): Label or "Start-of-Signal" token.
            hidden: (Optional torch.Tensor): RNN State vector
            add_sos (bool): Whether to add a zero vector at the begging as "start of sentence" token.
            batch_size: Batch size of the output tensor.

        Returns:
            g: (B, U, H) if add_sos is false, else (B, U + 1, H)
            hid: (h, c) where h is the final sequence hidden state and c is
                the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)
        """
        if isinstance(label, torch.Tensor):
            # label: [batch, 1]
            if label.dtype != torch.long:
                label = label.long()

        else:
            # Label is an integer
            if label == self._SOS:
                return self.decoder.predict(None, hidden, add_sos=add_sos, batch_size=batch_size)

            label = label_collate([[label]])

        # output: [B, 1, K]
        return self.decoder.predict(label, hidden, add_sos=add_sos, batch_size=batch_size)

    def _joint_step(self, enc, pred, log_normalize: Optional[bool] = None, language_ids=None): # CTEMO
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model. A torch.Tensor of shape [B, 1, H1]
            pred: Output of the Decoder model. A torch.Tensor of shape [B, 1, H2]
            log_normalize: Whether to log normalize or not. None will log normalize only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        with torch.no_grad():
            #Old
            # # logits = self.joint.joint(enc, pred)
            ## New for multisoftmax
            self.joint._fuse_loss_wer = False
            logits = self.joint(encoder_outputs=enc.transpose(1, 2), decoder_outputs=pred.transpose(1, 2), language_ids=language_ids)
            self.joint._fuse_loss_wer = True
            if log_normalize is None:
                if not logits.is_cuda:  # Use log softmax only if on CPU
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)
            else:
                if log_normalize:
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits

    def _joint_step_after_projection(self, enc, pred, log_normalize: Optional[bool] = None,language_ids=None) -> torch.Tensor:
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model after projection. A torch.Tensor of shape [B, 1, H]
            pred: Output of the Decoder model after projection. A torch.Tensor of shape [B, 1, H]
            log_normalize: Whether to log normalize or not. None will log normalize only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        with torch.no_grad():
            logits = self.joint.joint_after_projection(enc, pred,language_ids)

            if log_normalize is None:
                if not logits.is_cuda:  # Use log softmax only if on CPU
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)
            else:
                if log_normalize:
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits

class GreedyBatchedRNNTInfer(_GreedyRNNTInfer):
    """A batch level greedy transducer decoder.

    Batch level greedy decoding, performed auto-regressively.

    Args:
        decoder_model: AbstractRNNTDecoder implementation.
        joint_model: AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        preserve_alignments: Bool flag which preserves the history of alignments generated during
            greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of
            Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.
        preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores generated
            during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of List of floats.

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more confidence scores.
            U is the number of target tokens for the current timestep Ti.
        confidence_method_cfg: A dict-like object which contains the method name and settings to compute per-frame
            confidence scores.

            name: The method name (str).
                Supported values:
                    - 'max_prob' for using the maximum token probability as a confidence.
                    - 'entropy' for using a normalized entropy of a log-likelihood vector.

            entropy_type: Which type of entropy to use (str). Used if confidence_method_cfg.name is set to `entropy`.
                Supported values:
                    - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                        the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                        Note that for this entropy, the alpha should comply the following inequality:
                        (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                        where V is the model vocabulary size.
                    - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                        Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                        where α is a parameter. When α == 1, it works like the Gibbs entropy.
                        More: https://en.wikipedia.org/wiki/Tsallis_entropy
                    - 'renyi' for the Rényi entropy.
                        Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                        where α is a parameter. When α == 1, it works like the Gibbs entropy.
                        More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

            alpha: Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                When the alpha equals one, scaling is not applied to 'max_prob',
                and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

            entropy_norm: A mapping of the entropy value to the interval [0,1].
                Supported values:
                    - 'lin' for using the linear mapping.
                    - 'exp' for using exponential mapping with linear shift.
        loop_labels: Switching between decoding algorithms. Both algorithms produce equivalent results.
            loop_labels=True algorithm is faster (especially for large batches) but can use a bit more memory
            (negligible overhead compared to the amount of memory used by the encoder).
            loop_labels=False (default) is an implementation of a traditional decoding algorithm, which iterates over
            frames (encoder output vectors), and in the inner loop, decodes labels for the current frame one by one,
            stopping when <blank> is found.
            loop_labels=True iterates over labels, on each step finding the next non-blank label
            (evaluating Joint multiple times in inner loop); It uses a minimal possible amount of calls
            to prediction network (with maximum possible batch size),
            which makes it especially useful for scaling the prediction network.
    """

    def __init__(
        self,
        decoder_model: RNNTDecoder,
        joint_model: RNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
        preserve_frame_confidence: bool = False,
        confidence_method_cfg: Optional[DictConfig] = None,
        loop_labels: bool = False,
        use_cuda_graph_decoder: bool = False,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
            preserve_alignments=preserve_alignments,
            preserve_frame_confidence=preserve_frame_confidence,
            confidence_method_cfg=confidence_method_cfg,
        )

        self.use_cuda_graph_decoder = use_cuda_graph_decoder

        # Depending on availability of `blank_as_pad` support
        # switch between more efficient batch decoding technique
        self._decoding_computer = None
        if self.decoder.blank_as_pad:
            if loop_labels and use_cuda_graph_decoder:
                raise ValueError("loop_labels and use_cuda_graph_decoder is unsupported configuration")
            elif loop_labels:
                # default (faster) algo: loop over labels
                self._greedy_decode = self._greedy_decode_blank_as_pad_loop_labels
                self._decoding_computer = GreedyBatchedRNNTLoopLabelsComputer(
                    decoder=self.decoder,
                    joint=self.joint,
                    blank_index=self._blank_index,
                    max_symbols_per_step=self.max_symbols,
                    preserve_alignments=preserve_alignments,
                    preserve_frame_confidence=preserve_frame_confidence,
                    confidence_method_cfg=confidence_method_cfg,
                )
            elif use_cuda_graph_decoder:
                from nemo.collections.asr.parts.submodules.cuda_graph_rnnt_greedy_decoding import (
                    RNNTGreedyDecodeCudaGraph,
                )

                self._greedy_decode = RNNTGreedyDecodeCudaGraph(max_symbols_per_step, self)
            else:
                # previous algo: loop over frames
                self._greedy_decode = self._greedy_decode_blank_as_pad_loop_frames
        else:
            self._greedy_decode = self._greedy_decode_masked

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
        language_ids: Optional[List[str]] = None
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-regressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        return pack_hypotheses(self._greedy_decode(encoder_output, encoded_lengths, device=encoder_output.device, partial_hypotheses=partial_hypotheses, language_ids=language_ids), encoded_lengths)

    @torch.inference_mode()
    def _greedy_decode_blank_as_pad_loop_labels(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[list[Hypothesis]] = None,
    ) -> list[Hypothesis]:
        """
        Optimized batched greedy decoding.
        The main idea: search for next labels for the whole batch (evaluating Joint)
        and thus always evaluate prediction network with maximum possible batch size
        """
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not implemented")

        batched_hyps, alignments, last_decoder_state = self._decoding_computer(x=x, out_len=out_len)
        hyps = batched_hyps_to_hypotheses(batched_hyps, alignments)
        for hyp, state in zip(hyps, self.decoder.batch_split_states(last_decoder_state)):
            hyp.dec_state = state
        return hyps

    def _greedy_decode_masked(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
        language_ids=None,
    ):
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        # x: [B, T, D]
        # out_len: [B]
        # device: torch.device

        # Initialize state
        batchsize = x.shape[0]
        hypotheses = [
            Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)
        ]

        # Initialize Hidden state matrix (shared by entire batch)
        hidden = None

        # If alignments need to be preserved, register a danling list to hold the values
        if self.preserve_alignments:
            # alignments is a 3-dimensional dangling list representing B x T x U
            for hyp in hypotheses:
                hyp.alignments = [[]]
        else:
            alignments = None

        # If confidence scores need to be preserved, register a danling list to hold the values
        if self.preserve_frame_confidence:
            # frame_confidence is a 3-dimensional dangling list representing B x T x U
            for hyp in hypotheses:
                hyp.frame_confidence = [[]]

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long, device=device)
        last_label_without_blank = last_label.clone()

        # Mask buffers
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)
        blank_mask_prev = None

        # Get max sequence length
        max_out_len = out_len.max()

        with torch.inference_mode():
            for time_idx in range(max_out_len):
                f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

                # Prepare t timestamp batch variables
                not_blank = True
                symbols_added = 0

                # Reset blank mask
                blank_mask.mul_(False)

                # Update blank mask with time mask
                # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
                # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
                blank_mask = time_idx >= out_len
                blank_mask_prev = blank_mask.clone()

                # Start inner loop
                while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                    # Batch prediction and joint network steps
                    # If very first prediction step, submit SOS tag (blank) to pred_step.
                    # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                    if time_idx == 0 and symbols_added == 0 and hidden is None:
                        g, hidden_prime = self._pred_step(self._SOS, hidden, batch_size=batchsize)
                    else:
                        # Set a dummy label for the blank value
                        # This value will be overwritten by "blank" again the last label update below
                        # This is done as vocabulary of prediction network does not contain "blank" token of RNNT
                        last_label_without_blank_mask = last_label == self._blank_index
                        last_label_without_blank[last_label_without_blank_mask] = 0  # temp change of label
                        last_label_without_blank[~last_label_without_blank_mask] = last_label[
                            ~last_label_without_blank_mask
                        ]

                        # Perform batch step prediction of decoder, getting new states and scores ("g")
                        g, hidden_prime = self._pred_step(last_label_without_blank, hidden, batch_size=batchsize)

                    # Batched joint step - Output = [B, V + 1]
                    # If preserving per-frame confidence, log_normalize must be true
                    logp = self._joint_step(f, g, log_normalize=True if self.preserve_frame_confidence else None, language_ids=language_ids)[
                        :, 0, 0, :
                    ]

                    if logp.dtype != torch.float32:
                        logp = logp.float()

                    # Get index k, of max prob for batch
                    v, k = logp.max(1)
                    del g

                    # Update blank mask with current predicted blanks
                    # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                    k_is_blank = k == self._blank_index
                    blank_mask.bitwise_or_(k_is_blank)

                    # If preserving alignments, check if sequence length of sample has been reached
                    # before adding alignment
                    if self.preserve_alignments:
                        # Insert logprobs into last timestep per sample
                        logp_vals = logp.to('cpu')
                        logp_ids = logp_vals.max(1)[1]
                        for batch_idx, is_blank in enumerate(blank_mask):
                            # we only want to update non-blanks and first-time blanks,
                            # otherwise alignments will contain duplicate predictions
                            if time_idx < out_len[batch_idx] and (not blank_mask_prev[batch_idx] or not is_blank):
                                hypotheses[batch_idx].alignments[-1].append(
                                    (logp_vals[batch_idx], logp_ids[batch_idx])
                                )

                        del logp_vals

                    # If preserving per-frame confidence, check if sequence length of sample has been reached
                    # before adding confidence scores
                    if self.preserve_frame_confidence:
                        # Insert probabilities into last timestep per sample
                        confidence = self._get_confidence(logp)
                        for batch_idx, is_blank in enumerate(blank_mask):
                            if time_idx < out_len[batch_idx] and (not blank_mask_prev[batch_idx] or not is_blank):
                                hypotheses[batch_idx].frame_confidence[-1].append(confidence[batch_idx])
                    del logp

                    blank_mask_prev.bitwise_or_(blank_mask)

                    # If all samples predict / have predicted prior blanks, exit loop early
                    # This is equivalent to if single sample predicted k
                    if blank_mask.all():
                        not_blank = False
                    else:
                        # Collect batch indices where blanks occurred now/past
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)

                        # Recover prior state for all samples which predicted blank now/past
                        if hidden is not None:
                            # LSTM has 2 states
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, hidden, blank_indices)

                        elif len(blank_indices) > 0 and hidden is None:
                            # Reset state if there were some blank and other non-blank predictions in batch
                            # Original state is filled with zeros so we just multiply
                            # LSTM has 2 states
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, None, blank_indices, value=0.0)

                        # Recover prior predicted label for all samples which predicted blank now/past
                        k[blank_indices] = last_label[blank_indices, 0]

                        # Update new label and hidden state for next iteration
                        last_label = k.view(-1, 1)
                        hidden = hidden_prime

                        # Update predicted labels, accounting for time mask
                        # If blank was predicted even once, now or in the past,
                        # Force the current predicted label to also be blank
                        # This ensures that blanks propogate across all timesteps
                        # once they have occured (normally stopping condition of sample level loop).
                        for kidx, ki in enumerate(k):
                            if blank_mask[kidx] == 0:
                                hypotheses[kidx].y_sequence.append(ki)
                                hypotheses[kidx].timestep.append(time_idx)
                                hypotheses[kidx].score += float(v[kidx])

                    symbols_added += 1

                # If preserving alignments, convert the current Uj alignments into a torch.Tensor
                # Then preserve U at current timestep Ti
                # Finally, forward the timestep history to Ti+1 for that sample
                # All of this should only be done iff the current time index <= sample-level AM length.
                # Otherwise ignore and move to next sample / next timestep.
                if self.preserve_alignments:

                    # convert Ti-th logits into a torch array
                    for batch_idx in range(batchsize):

                        # this checks if current timestep <= sample-level AM length
                        # If current timestep > sample-level AM length, no alignments will be added
                        # Therefore the list of Uj alignments is empty here.
                        if len(hypotheses[batch_idx].alignments[-1]) > 0:
                            hypotheses[batch_idx].alignments.append([])  # blank buffer for next timestep

                # Do the same if preserving per-frame confidence
                if self.preserve_frame_confidence:

                    for batch_idx in range(batchsize):
                        if len(hypotheses[batch_idx].frame_confidence[-1]) > 0:
                            hypotheses[batch_idx].frame_confidence.append([])  # blank buffer for next timestep

        # Remove trailing empty list of alignments at T_{am-len} x Uj
        if self.preserve_alignments:
            for batch_idx in range(batchsize):
                if len(hypotheses[batch_idx].alignments[-1]) == 0:
                    del hypotheses[batch_idx].alignments[-1]

        # Remove trailing empty list of confidence scores at T_{am-len} x Uj
        if self.preserve_frame_confidence:
            for batch_idx in range(batchsize):
                if len(hypotheses[batch_idx].frame_confidence[-1]) == 0:
                    del hypotheses[batch_idx].frame_confidence[-1]

        # Preserve states
        for batch_idx in range(batchsize):
            hypotheses[batch_idx].dec_state = self.decoder.batch_select_state(hidden, batch_idx)

        return hypotheses

    def _greedy_decode_blank_as_pad_loop_frames(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
        language_ids=None,
    ):
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        with torch.inference_mode():
            # x: [B, T, D]
            # out_len: [B]
            # device: torch.device

            # Initialize list of Hypothesis
            batchsize = x.shape[0]
            hypotheses = [
                Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)
            ]

            # Initialize Hidden state matrix (shared by entire batch)
            hidden = None

            # If alignments need to be preserved, register a dangling list to hold the values
            if self.preserve_alignments:
                # alignments is a 3-dimensional dangling list representing B x T x U
                for hyp in hypotheses:
                    hyp.alignments = [[]]

            # If confidence scores need to be preserved, register a dangling list to hold the values
            if self.preserve_frame_confidence:
                # frame_confidence is a 3-dimensional dangling list representing B x T x U
                for hyp in hypotheses:
                    hyp.frame_confidence = [[]]

            # Last Label buffer + Last Label without blank buffer
            # batch level equivalent of the last_label
            last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long, device=device)

            # Mask buffers
            blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)
            blank_mask_prev = None

            # Get max sequence length
            max_out_len = out_len.max()
            for time_idx in range(max_out_len):
                f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

                # Prepare t timestamp batch variables
                not_blank = True
                symbols_added = 0

                # Reset blank mask
                blank_mask.mul_(False)

                # Update blank mask with time mask
                # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
                # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
                blank_mask = time_idx >= out_len
                blank_mask_prev = blank_mask.clone()

                # Start inner loop
                while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                    # Batch prediction and joint network steps
                    # If very first prediction step, submit SOS tag (blank) to pred_step.
                    # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                    if time_idx == 0 and symbols_added == 0 and hidden is None:
                        g, hidden_prime = self._pred_step(self._SOS, hidden, batch_size=batchsize)
                    else:
                        # Perform batch step prediction of decoder, getting new states and scores ("g")
                        g, hidden_prime = self._pred_step(last_label, hidden, batch_size=batchsize)

                    # Batched joint step - Output = [B, V + 1]
                    # If preserving per-frame confidence, log_normalize must be true
                    logp = self._joint_step(f, g, log_normalize=True if self.preserve_frame_confidence else None, language_ids=language_ids)[
                        :, 0, 0, :
                    ]

                    if logp.dtype != torch.float32:
                        logp = logp.float()

                    # Get index k, of max prob for batch
                    v, k = logp.max(1)
                    del g

                    # Update blank mask with current predicted blanks
                    # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                    k_is_blank = k == self._blank_index
                    blank_mask.bitwise_or_(k_is_blank)

                    del k_is_blank

                    # If preserving alignments, check if sequence length of sample has been reached
                    # before adding alignment
                    if self.preserve_alignments:
                        # Insert logprobs into last timestep per sample
                        logp_vals = logp.to('cpu')
                        logp_ids = logp_vals.max(1)[1]
                        for batch_idx, is_blank in enumerate(blank_mask):
                            # we only want to update non-blanks and first-time blanks,
                            # otherwise alignments will contain duplicate predictions
                            if time_idx < out_len[batch_idx] and (not blank_mask_prev[batch_idx] or not is_blank):
                                hypotheses[batch_idx].alignments[-1].append(
                                    (logp_vals[batch_idx], logp_ids[batch_idx])
                                )
                        del logp_vals

                    # If preserving per-frame confidence, check if sequence length of sample has been reached
                    # before adding confidence scores
                    if self.preserve_frame_confidence:
                        # Insert probabilities into last timestep per sample
                        confidence = self._get_confidence(logp)
                        for batch_idx, is_blank in enumerate(blank_mask):
                            if time_idx < out_len[batch_idx] and (not blank_mask_prev[batch_idx] or not is_blank):
                                hypotheses[batch_idx].frame_confidence[-1].append(confidence[batch_idx])
                    del logp

                    blank_mask_prev.bitwise_or_(blank_mask)

                    # If all samples predict / have predicted prior blanks, exit loop early
                    # This is equivalent to if single sample predicted k
                    if blank_mask.all():
                        not_blank = False
                    else:
                        # Collect batch indices where blanks occurred now/past
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)

                        # Recover prior state for all samples which predicted blank now/past
                        if hidden is not None:
                            # LSTM has 2 states
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, hidden, blank_indices)

                        elif len(blank_indices) > 0 and hidden is None:
                            # Reset state if there were some blank and other non-blank predictions in batch
                            # Original state is filled with zeros so we just multiply
                            # LSTM has 2 states
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, None, blank_indices, value=0.0)

                        # Recover prior predicted label for all samples which predicted blank now/past
                        k[blank_indices] = last_label[blank_indices, 0]

                        # Update new label and hidden state for next iteration
                        last_label = k.clone().view(-1, 1)
                        hidden = hidden_prime

                        # Update predicted labels, accounting for time mask
                        # If blank was predicted even once, now or in the past,
                        # Force the current predicted label to also be blank
                        # This ensures that blanks propogate across all timesteps
                        # once they have occured (normally stopping condition of sample level loop).
                        for kidx, ki in enumerate(k):
                            if blank_mask[kidx] == 0:
                                hypotheses[kidx].y_sequence.append(ki)
                                hypotheses[kidx].timestep.append(time_idx)
                                hypotheses[kidx].score += float(v[kidx])
                        symbols_added += 1

                # If preserving alignments, convert the current Uj alignments into a torch.Tensor
                # Then preserve U at current timestep Ti
                # Finally, forward the timestep history to Ti+1 for that sample
                # All of this should only be done iff the current time index <= sample-level AM length.
                # Otherwise ignore and move to next sample / next timestep.
                if self.preserve_alignments:

                    # convert Ti-th logits into a torch array
                    for batch_idx in range(batchsize):

                        # this checks if current timestep <= sample-level AM length
                        # If current timestep > sample-level AM length, no alignments will be added
                        # Therefore the list of Uj alignments is empty here.
                        if len(hypotheses[batch_idx].alignments[-1]) > 0:
                            hypotheses[batch_idx].alignments.append([])  # blank buffer for next timestep

                # Do the same if preserving per-frame confidence
                if self.preserve_frame_confidence:

                    for batch_idx in range(batchsize):
                        if len(hypotheses[batch_idx].frame_confidence[-1]) > 0:
                            hypotheses[batch_idx].frame_confidence.append([])  # blank buffer for next timestep

            # Remove trailing empty list of alignments at T_{am-len} x Uj
            if self.preserve_alignments:
                for batch_idx in range(batchsize):
                    if len(hypotheses[batch_idx].alignments[-1]) == 0:
                        del hypotheses[batch_idx].alignments[-1]

            # Remove trailing empty list of confidence scores at T_{am-len} x Uj
            if self.preserve_frame_confidence:
                for batch_idx in range(batchsize):
                    if len(hypotheses[batch_idx].frame_confidence[-1]) == 0:
                        del hypotheses[batch_idx].frame_confidence[-1]

        # Preserve states
        for batch_idx in range(batchsize):
            hypotheses[batch_idx].dec_state = self.decoder.batch_select_state(hidden, batch_idx)

        return hypotheses

def batched_hyps_to_hypotheses(
    batched_hyps: BatchedHyps, alignments: Optional[BatchedAlignments] = None
) -> List[Hypothesis]:
    """
    Convert batched hypotheses to a list of Hypothesis objects.
    Keep this function separate to allow for jit compilation for BatchedHyps class (see tests)

    Args:
        batched_hyps: BatchedHyps object
        alignments: BatchedAlignments object, optional; must correspond to BatchedHyps if present

    Returns:
        list of Hypothesis objects
    """
    hypotheses = [
        Hypothesis(
            score=batched_hyps.scores[i].item(),
            y_sequence=batched_hyps.transcript[i, : batched_hyps.current_lengths[i]],
            timestep=batched_hyps.timesteps[i, : batched_hyps.current_lengths[i]],
            alignments=None,
            dec_state=None,
        )
        for i in range(batched_hyps.scores.shape[0])
    ]
    if alignments is not None:
        # move all data to cpu to avoid overhead with moving data by chunks
        alignment_lengths = alignments.current_lengths.cpu().tolist()
        if alignments.with_alignments:
            alignment_logits = alignments.logits.cpu()
            alignment_labels = alignments.labels.cpu()
        if alignments.with_frame_confidence:
            frame_confidence = alignments.frame_confidence.cpu()

        # for each hypothesis - aggregate alignment using unique_consecutive for time indices (~itertools.groupby)
        for i in range(len(hypotheses)):
            hypotheses[i].alignments = []
            if alignments.with_frame_confidence:
                hypotheses[i].frame_confidence = []
            _, grouped_counts = torch.unique_consecutive(
                alignments.timesteps[i, : alignment_lengths[i]], return_counts=True
            )
            start = 0
            for timestep_cnt in grouped_counts.tolist():
                if alignments.with_alignments:
                    hypotheses[i].alignments.append(
                        [(alignment_logits[i, start + j], alignment_labels[i, start + j]) for j in range(timestep_cnt)]
                    )
                if alignments.with_frame_confidence:
                    hypotheses[i].frame_confidence.append(
                        [frame_confidence[i, start + j] for j in range(timestep_cnt)]
                    )
                start += timestep_cnt
    return hypotheses
