import logging
import numpy as np
import torch
import torch.nn as nn
import logging
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
BLANK_ID = 256
CONSTANT = 1e-5

def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
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
        subsampling_conv_chunking_factor=1,
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
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor
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
        self.conv2d_subsampling = True
        self.conv = nn.Sequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def forward(self, x:torch.Tensor, lengths:torch.Tensor):

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
                    x = self.conv_split_by_channel(x)
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

        return x, calc_length(lengths,self._left_padding + self._right_padding,self._kernel_size,self._stride,self._ceil_mode,self._sampling_num,)

    def reset_parameters(self):
        # initialize weights
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

def make_chunks(file_path,aggressiveness=2,min_chunk_sec=10,max_chunk_sec=15,frame_ms=30,format=None):
    wav, sr = torchaudio.load(file_path, normalize=True,format=format)
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
    def __init__(self, audio_path):
        self.data,self.ts = make_chunks(audio_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx],self.ts[idx]
