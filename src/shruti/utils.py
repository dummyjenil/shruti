import numpy as np
import torch
import torch.nn as nn
import librosa
import torch.nn.functional as F
import webrtcvad
from datetime import timedelta
import srt
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Hypothesis:
    y_sequence: List[int] = field(default_factory=list)
    timestep: List[int] = field(default_factory=list)

def label_collate(labels):
    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")
    batch_size = len(labels)
    max_len = max(len(label) for label in labels)
    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        if isinstance(l, torch.Tensor):
            l = l.squeeze().cpu().numpy()
            if l.ndim == 0:
                l = [l.item()]
        cat_labels[e, : len(l)] = l
    return torch.tensor(cat_labels, dtype=torch.int64)

class RNNTDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size + 1, hidden_dim,padding_idx=vocab_size)
        self.lstm = nn.LSTM(hidden_dim,hidden_dim,num_layers,batch_first=True)
    def forward(self,y: torch.Tensor,state: Optional[Tuple[torch.Tensor]]):
        return self.lstm(self.embed(y), state)

class RNNTJoint(nn.Module):
    def __init__(self, enc_dim, pred_dim, joint_dim, vocab_size):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)
        self.relu = nn.ReLU(True)
        self.joint_net = nn.Linear(joint_dim, vocab_size)

    def forward(self, enc, pred):
        return self.joint_net(self.relu(self.enc_proj(enc) + self.pred_proj(pred)))

def calc_length(lengths, all_paddings, kernel_size, stride, repeat_num=1):
    add_pad = all_paddings - kernel_size
    for _ in range(repeat_num):
        lengths = torch.floor(torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + 1.0)
    return lengths

class MelPreprocessor(nn.Module):
    def __init__(
        self,
        win_length=400,
        hop_length=160,
        n_fft=512,
        preemph=0.97,
        log_zero_guard_value=2 ** -24,
        pad_to=16,
        mag_power=2.0,
    ):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.preemph = preemph
        self.log_zero_guard_value = log_zero_guard_value
        self.pad_to = pad_to
        self.mag_power = mag_power
        self.constant = 1e-5
        self.sr = 16000
        self.window = nn.Parameter(torch.hann_window(self.win_length, periodic=False))
        self.mel_fb = nn.Parameter(torch.tensor(librosa.filters.mel(sr=self.sr,n_fft=n_fft,n_mels=80), dtype=torch.float32))

    def forward(self, x, lengths):
        """
        x: [B, T]
        lengths:  [B]
        returns:
            x: [B, n_mels, T']
            lengths: [B]
        """
        x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
        x = torch.stft(x,self.n_fft,self.hop_length,self.win_length,self.window.to(dtype=x.dtype),return_complex=True)
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + self.constant)
        x = x.pow(self.mag_power)
        x = torch.matmul(self.mel_fb.to(x.dtype), x)
        x = torch.log(x + self.log_zero_guard_value)
        lengths = (torch.floor_divide(lengths, self.hop_length) + 1).to(torch.long)
        B, C, T = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T) >= lengths.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1), 0.0)
        mean = x.sum(-1) / lengths.unsqueeze(1)
        var = ((x - mean.unsqueeze(-1)) ** 2).sum(-1) / (lengths.unsqueeze(1) - 1)
        std = torch.sqrt(var + self.constant)
        x = (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        x = x.masked_fill(mask.unsqueeze(1), 0.0)
        if self.pad_to > 0:
            pad_amt = x.shape[-1] % self.pad_to
            if pad_amt != 0:
                x = F.pad(x,(0, self.pad_to - pad_amt),value=0.0)
        return x, lengths

class ConvSubsampling(nn.Module):
    def __init__(
        self,
        sampling_num,
        feat_out,
        conv_channels,
        multilingual=True
    ):
        super().__init__()
        self._sampling_num = sampling_num
        in_channels = 1
        layers = []
        if multilingual:
            layers.extend([nn.Conv2d(in_channels,conv_channels,3,2,1),nn.ReLU(True)])
            in_channels = conv_channels
            for _ in range(sampling_num - 1):
                layers.append(nn.Conv2d(in_channels,in_channels,3,2,1,groups=in_channels,))
                layers.append(nn.Conv2d(in_channels,conv_channels,1))
                layers.append(nn.ReLU(True))
                in_channels = conv_channels
        else:
            for _ in range(self._sampling_num):
                layers.append(nn.Conv2d(in_channels,conv_channels,3,2,1))
                layers.append(nn.ReLU(True))
                in_channels = conv_channels
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(conv_channels * int(calc_length(torch.tensor(80, dtype=torch.float),2,3,2,self._sampling_num)), feat_out)

    def forward(self, x:torch.Tensor):
        x = self.conv(x.transpose(1, 2).unsqueeze(1))
        b, c, t, f = x.size()
        return self.out(x.transpose(1, 2).reshape(b, t, -1))

class MaskPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad_mask = None
    def forward(self, hidden_states):
        return hidden_states.float().masked_fill(self.pad_mask.unsqueeze(1), 0.0)

def make_chunks(wav,sr,aggressiveness=2,min_chunk_sec=10,max_chunk_sec=15,frame_ms=30):
    wav = wav.mean(0, keepdim=True)
    wav_int16 = (wav * 32768).clamp(-32768, 32767).short().squeeze(0)
    frame_len = int(sr * frame_ms / 1000)
    total_frames = len(wav_int16) // frame_len
    wav_int16 = wav_int16[: total_frames * frame_len]
    total_samples = len(wav_int16)
    frames = wav_int16.view(total_frames, frame_len)

    vad = webrtcvad.Vad(aggressiveness)
    is_speech = torch.zeros(total_frames, dtype=torch.bool)

    for i, f in enumerate(frames):
        try:
            is_speech[i] = vad.is_speech(f.cpu().numpy().tobytes(), sr)
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
        starts = s.cpu() + np.array(hyp.timestep) * denormalizer
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
    return torch.stack([F.pad(x, (0, max_len - len(x))) for x in audios]), lengths, torch.stack(times)

class ChunkedData(Dataset):
    def __init__(self, wav,sr):
        self.data,self.ts = make_chunks(wav,sr)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx],self.ts[idx]

class GreedyRNNTInfer(nn.Module):
    def __init__(
        self,
        decoder: RNNTDecoder,
        joint: RNNTJoint,
        blank_id: int,
        max_symbols_per_step: int = 10,
    ):
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        self.blank_id = blank_id
        self.max_symbols = max_symbols_per_step

    def forward(self, x, out_len):
        B, T, _ = x.shape
        device = x.device

        hidden = None
        last_label = torch.full((B, 1), self.blank_id, device=device)

        hypotheses = [Hypothesis() for _ in range(B)]
        
        for t in range(T):
            enc_t = x[:, t:t+1, :]

            valid = t < out_len  # (B,)
            if not valid.any():
                break

            symbols_added = 0
            while symbols_added < self.max_symbols:
                hidden_old = hidden
                
                pred, hidden_new = self.decoder(last_label, hidden)
                logits = self.joint(enc_t, pred)
                logp = torch.log_softmax(logits.squeeze(1), dim=-1)
                next_label = logp.argmax(dim=-1)

                # Force blank for invalid frames
                next_label = torch.where(
                    valid,
                    next_label,
                    torch.full_like(next_label, self.blank_id)
                )

                is_blank = next_label == self.blank_id
                if is_blank.all():
                    break

                # Append to hypotheses
                for b in range(B):
                    if valid[b] and not is_blank[b]:
                        hypotheses[b].y_sequence.append(int(next_label[b]))
                        hypotheses[b].timestep.append(t)

                # Update last_label for non-blank predictions
                non_blank = ~is_blank & valid
                last_label = torch.where(
                    non_blank.unsqueeze(1),
                    next_label.unsqueeze(1),
                    last_label
                )
                
                # Update hidden states - keep old states for blank predictions
                if hidden_old is None:
                    hidden = hidden_new
                else:
                    blank_mask = is_blank | ~valid
                    hidden = (
                        torch.where(blank_mask.view(1, -1, 1), hidden_old[0], hidden_new[0]),
                        torch.where(blank_mask.view(1, -1, 1), hidden_old[1], hidden_new[1])
                    )
                symbols_added += 1
        return hypotheses
