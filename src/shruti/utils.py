import numpy as np
import torch
import torch.nn as nn
import librosa
import torch.nn.functional as F
import torchaudio
import webrtcvad
from datetime import timedelta
import srt
from torch.utils.data import Dataset
import torchaudio.functional as F_torchaudio
import math
from shruti.rnnt_utils import Hypothesis
from shruti.spleeter import Spleeter
CONSTANT = 1e-5

def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for _ in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)

class MelPreprocessor(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=512,
        n_mels=80,
        preemph=0.97,
        log=True,
        log_zero_guard_value=2 ** -24,
        normalize="per_feature",
        pad_to=16,
        pad_value=0.0,
        mag_power=2.0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.preemph = preemph
        self.log = log
        self.log_zero_guard_value = log_zero_guard_value
        self.normalize = normalize
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.mag_power = mag_power

        # Window (NeMo uses periodic=False)
        window = torch.hann_window(self.win_length, periodic=False)
        self.register_buffer("window", window)

        # Precomputed NeMo mel filterbank (librosa-based)
        mel_fb = torch.tensor(librosa.filters.mel(sr=sample_rate,n_fft=n_fft,n_mels=n_mels), dtype=torch.float32)
        self.register_buffer("mel_fb", mel_fb)

    def _preemphasis(self, x):
        if self.preemph == 0.0:
            return x
        return torch.cat(
            (x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1
        )

    def _get_feat_len(self, lengths):
        # NeMo formula (center=True)
        pad = self.n_fft
        return torch.floor_divide(lengths + pad - self.n_fft, self.hop_length) + 1

    def forward(self, waveform, lengths):
        """
        waveform: [B, T]
        lengths:  [B]
        returns:
            feats: [B, n_mels, T']
            feat_lens: [B]
        """

        # Preemphasis
        x = self._preemphasis(waveform)

        # STFT (NeMo-style)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=x.dtype),
            center=True,
            return_complex=True,
        )

        # Magnitude
        spec = torch.view_as_real(spec)
        mag = torch.sqrt(spec.pow(2).sum(-1) + CONSTANT)

        # Power
        if self.mag_power != 1.0:
            mag = mag.pow(self.mag_power)

        # Mel projection (NeMo order)
        feats = torch.matmul(self.mel_fb.to(mag.dtype), mag)

        # Log
        if self.log:
            feats = torch.log(feats + self.log_zero_guard_value)

        # Feature lengths
        feat_lens = self._get_feat_len(lengths).to(torch.long)

        # Normalization (NeMo-style)
        if self.normalize == "per_feature":
            B, C, T = feats.shape
            mask = (
                torch.arange(T, device=feats.device)
                .unsqueeze(0)
                .expand(B, T)
                >= feat_lens.unsqueeze(1)
            )

            feats = feats.masked_fill(mask.unsqueeze(1), 0.0)
            mean = feats.sum(-1) / feat_lens.unsqueeze(1)
            var = ((feats - mean.unsqueeze(-1)) ** 2).sum(-1) / (feat_lens.unsqueeze(1) - 1)
            std = torch.sqrt(var + CONSTANT)

            feats = (feats - mean.unsqueeze(-1)) / std.unsqueeze(-1)
            feats = feats.masked_fill(mask.unsqueeze(1), 0.0)

        # Pad to multiple of pad_to
        if self.pad_to > 0:
            pad_amt = feats.shape[-1] % self.pad_to
            if pad_amt != 0:
                feats = F.pad(
                    feats,
                    (0, self.pad_to - pad_amt),
                    value=self.pad_value,
                )

        return feats, feat_lens

class ConvSubsampling(nn.Module):

    def __init__(
        self,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        activation=nn.ReLU(True),
    ):
        super().__init__()
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        in_channels = 1
        layers = []
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False
        self._left_padding = (self._kernel_size - 1) // 2
        self._right_padding = (self._kernel_size - 1) // 2
        self._max_cache_len = 0
        layers.append(nn.Conv2d(in_channels,conv_channels,self._kernel_size,self._stride,self._left_padding))
        in_channels = conv_channels
        layers.append(activation)
        for _ in range(self._sampling_num - 1):
            layers.extend([nn.Conv2d(in_channels,in_channels,self._kernel_size,self._stride,self._left_padding,groups=in_channels,),nn.Conv2d(in_channels,conv_channels,1,1),activation])
            in_channels = conv_channels
        self.out = nn.Linear(conv_channels * int(calc_length(
            lengths=torch.tensor(feat_in, dtype=torch.float),
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )), feat_out)
        self.conv = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor, lengths:torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, calc_length(lengths,self._left_padding + self._right_padding,self._kernel_size,self._stride,self._ceil_mode,self._sampling_num,)

def make_chunks(wav,sr,aggressiveness=2,min_chunk_sec=10,max_chunk_sec=15,frame_ms=30):
   
    wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = F_torchaudio.resample(wav, sr, 16000)
        sr = 16000

    wav_int16 = (wav * 32768).clamp(-32768, 32767).short().squeeze(0)
    total_samples = len(wav_int16)

    frame_len = int(sr * frame_ms / 1000)
    total_frames = len(wav_int16) // frame_len
    wav_int16 = wav_int16[: total_frames * frame_len]
    frames = wav_int16.view(total_frames, frame_len)

    vad = webrtcvad.Vad(aggressiveness)
    is_speech = torch.zeros(total_frames, dtype=torch.bool)

    for i, f in enumerate(frames):
        try:
            is_speech[i] = vad.is_speech(f.numpy().tobytes(), sr)
        except:
            is_speech[i] = False

    segs, start_idx = [], None
    for i, s in enumerate(is_speech):
        if s and start_idx is None:
            start_idx = i
        elif not s and start_idx is not None:
            segs.append((start_idx, i))
            start_idx = None
    if start_idx is not None:
        segs.append((start_idx, len(is_speech)))
    chunks = []
    times_list = []
    start_sample = 0
    min_len = int(min_chunk_sec * sr)
    max_len = int(max_chunk_sec * sr)
    while start_sample < total_samples:
        end_sample = min(start_sample + max_len, total_samples)
        chunk_end_frame = end_sample // frame_len
        while chunk_end_frame < len(is_speech) and is_speech[chunk_end_frame]:
            chunk_end_frame += 1
            end_sample = min(chunk_end_frame * frame_len, total_samples)
            if end_sample - start_sample > max_len * 1.5:
                break
        if end_sample - start_sample < min_len and end_sample < total_samples:
            end_sample = min(start_sample + min_len, total_samples)
        chunk = wav[:, start_sample:end_sample]
        chunks.append(chunk.squeeze())
        times_list.append([
            round(start_sample / sr, 2),
            round(end_sample / sr, 2)
        ])
        start_sample = end_sample
    times = torch.tensor(times_list, dtype=torch.float32)
    return chunks, times

def make_srt(h: list[Hypothesis], ts: torch.Tensor, tokenizer, denormalizer: int=0.08): # 0.08 is making because of subsampling_factor * window stride = 8 * 0.01
    timestamp = []
    for hyp, (s, e) in zip(h, ts):
        starts = s + np.array(hyp.timestep) * denormalizer
        for y, st, en in zip(hyp.y_sequence,starts,list(starts[1:]) + [e]):
            timestamp.append({"text": tokenizer[int(y)],"start": float(st),"end": float(en)})
        timestamp.append({"text": "<line>","start": float(e),"end": float(e + 0.005)})
    return [
        srt.Subtitle(
            index,
            timedelta(seconds=item["start"]),
            timedelta(seconds=item["end"]),
            item["text"]
        )
        for index, item in enumerate(timestamp, 1)
    ]

def padding_audio(batch):
    audios, times = zip(*batch)

    lengths = torch.tensor([len(x) for x in audios])
    max_len = lengths.max()

    padded = torch.stack([
        F.pad(x, (0, max_len - len(x)))
        for x in audios
    ])

    times = torch.stack(times)   # each is a tensor [start, end]

    return padded, lengths, times

class ChunkedData(Dataset):
    def __init__(self, audio_path,spleeter:Spleeter):
        wav, sr = torchaudio.load(audio_path)
        stem = spleeter.separate_audio_in_chunks(wav,sr)
        # self.music = stem['instruments']
        self.vocal = stem['vocal']
        del stem
        self.data,self.ts = make_chunks(self.vocal,44100)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx],self.ts[idx]


class MaskPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad_mask = None
    def forward(self, hidden_states):
        if self.pad_mask is None:
            return hidden_states
        return hidden_states.float().masked_fill(self.pad_mask.unsqueeze(1), 0.0)
