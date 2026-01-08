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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

@dataclass
class Hypothesis:
    score:float
    y_sequence: List[int] = field(default_factory=list)
    timestep: List[int] = field(default_factory=list)
    dec_state: Optional[Tuple] = None

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
    def __init__(self, pred_hidden, pred_rnn_layers, vocab_size):
        super().__init__()
        self.pred_hidden = pred_hidden
        self.pred_rnn_layers = pred_rnn_layers
        self.embed = nn.Embedding(vocab_size + 1, pred_hidden, padding_idx=vocab_size)
        self.lstm = nn.LSTM(pred_hidden,pred_hidden,pred_rnn_layers)
    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype
        if y is not None:
            if y.device != device:
                y = y.to(device)
            y = self.embed(y)
        else:
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size
            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  
        else:
            start = None  
        y = y.transpose(0, 1)  
        g, hid = self.lstm(y, state)
        g = g.transpose(0, 1)  
        del y, start, state
        return g, hid

    def batch_copy_states(
        self,
        old_states: List[torch.Tensor],
        new_states: List[torch.Tensor],
        ids: List[int],
        value: Optional[float] = None,
    ) -> List[torch.Tensor]:
        for state_id in range(len(old_states)):
            if value is None:
                old_states[state_id][:, ids, :] = new_states[state_id][:, ids, :]
            else:
                old_states[state_id][:, ids, :] *= 0.0
                old_states[state_id][:, ids, :] += value
        return old_states

class RNNTJoint(nn.Module):
    def __init__(self, pred_hidden, enc_hidden, joint_hidden, vocab_size):
        super().__init__()
        self.pred = nn.Linear(pred_hidden, joint_hidden)
        self.enc = nn.Linear(enc_hidden, joint_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.joint_net = nn.Linear(joint_hidden, vocab_size + 1)
        self.temperature = 1.0
    def forward(self, g, f):
        x = self.joint_net(self.relu(self.enc(g).unsqueeze(2) + self.pred(f).unsqueeze(1)))
        if not x.is_cuda:  
            if self.temperature != 1.0:
                x = (x / self.temperature).log_softmax(dim=-1)
            else:
                x = x.log_softmax(dim=-1)
        return x

def calc_length(lengths, all_paddings, kernel_size, stride, repeat_num=1):
    add_pad = all_paddings - kernel_size
    for _ in range(repeat_num):
        lengths = torch.floor(torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + 1.0)
    return lengths

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
        self.constant = 1e-5
        self.window = nn.Parameter(torch.hann_window(self.win_length, periodic=False))
        self.mel_fb = nn.Parameter(torch.tensor(librosa.filters.mel(sr=sample_rate,n_fft=n_fft,n_mels=n_mels), dtype=torch.float32))

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
        mag = torch.sqrt(spec.pow(2).sum(-1) + self.constant)

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
            std = torch.sqrt(var + self.constant)

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
        conv_channels
    ):
        super().__init__()
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        in_channels = 1
        layers = []
        self._stride = 2
        self._kernel_size = 3
        self._left_padding = (self._kernel_size - 1) // 2
        self._right_padding = (self._kernel_size - 1) // 2
        layers.append(nn.Conv2d(in_channels,conv_channels,self._kernel_size,self._stride,self._left_padding))
        in_channels = conv_channels
        layers.append(nn.ReLU(True))
        for _ in range(self._sampling_num - 1):
            layers.extend([nn.Conv2d(in_channels,in_channels,self._kernel_size,self._stride,self._left_padding,groups=in_channels,),
                           nn.Conv2d(in_channels,conv_channels,1,1),
                           nn.ReLU(True)])
            in_channels = conv_channels
        self.out = nn.Linear(conv_channels * int(calc_length(torch.tensor(feat_in, dtype=torch.float),self._left_padding + self._right_padding,self._kernel_size,self._stride,repeat_num=self._sampling_num)), feat_out)
        self.conv = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor, lengths:torch.Tensor):
        x = self.conv(x.transpose(1, 2).unsqueeze(1))
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, calc_length(lengths,self._left_padding + self._right_padding,self._kernel_size,self._stride,self._sampling_num,).to(torch.int64)

class MaskPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad_mask = None
    def forward(self, hidden_states):
        if self.pad_mask is None:
            return hidden_states
        return hidden_states.float().masked_fill(self.pad_mask.unsqueeze(1), 0.0)

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

    padded = torch.stack([
        F.pad(x, (0, max_len - len(x)))
        for x in audios
    ])

    times = torch.stack(times)   # each is a tensor [start, end]

    return padded, lengths, times

class ChunkedData(Dataset):
    def __init__(self, audio_path,spleeter:nn.Module,batch_size=2,chunk_duration_sec=30):
        wav, sr = torchaudio.load(audio_path)
        if spleeter:
            stem = spleeter.separate_audio_in_chunks(wav,sr,batch_size=batch_size,chunk_duration_sec=chunk_duration_sec)
            # self.music = stem['instruments']
            self.vocal = torchaudio.functional.resample(stem['vocal'], 44100, 16000)
            del stem
        else:
            self.vocal = torchaudio.functional.resample(wav, sr, 16000)

        self.data,self.ts = make_chunks(self.vocal,16000)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx],self.ts[idx]


class GreedyRNNTInfer(nn.Module):
    def __init__(
        self,
        pred_hidden,pred_rnn_layers,vocab_size,lang_vocab_size,d_model,joint_hidden,blank_id,max_symbols):
        super().__init__()
        self.decoder = RNNTDecoder(pred_hidden,pred_rnn_layers,vocab_size)
        self.joint = RNNTJoint(pred_hidden,d_model,joint_hidden,lang_vocab_size)
        self._blank_index = blank_id
        self._SOS = blank_id  
        self.max_symbols = max_symbols
    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        hidden: Optional[torch.Tensor],
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(label, torch.Tensor):
            if label.dtype != torch.long:
                label = label.long()
        else:
            if label == self._SOS:
                return self.decoder.predict(None, hidden, add_sos=add_sos, batch_size=batch_size)
            label = label_collate([[label]])
        return self.decoder.predict(label, hidden, add_sos=add_sos, batch_size=batch_size)

    def _joint_step(self, enc, pred, log_normalize: Optional[bool] = None, language_ids=None): 
        logits = self.joint(enc, pred)
        if not logits.is_cuda:  
            logits = logits.log_softmax(dim=len(logits.shape) - 1)
        return logits

    def forward(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        language_ids=None,
    ):
        with torch.inference_mode():
            batchsize = x.shape[0]
            hypotheses = [
                Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)
            ]
            hidden = None
            last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long, device=device)
            blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)
            blank_mask_prev = None
            max_out_len = out_len.max()
            for time_idx in range(max_out_len):
                f = x.narrow(dim=1, start=time_idx, length=1)  
                not_blank = True
                symbols_added = 0
                blank_mask.mul_(False)
                blank_mask = time_idx >= out_len
                blank_mask_prev = blank_mask.clone()
                while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                    if time_idx == 0 and symbols_added == 0 and hidden is None:
                        g, hidden_prime = self._pred_step(self._SOS, hidden, batch_size=batchsize)
                    else:
                        g, hidden_prime = self._pred_step(last_label, hidden, batch_size=batchsize)
                    logp = self._joint_step(f, g, log_normalize=None, language_ids=language_ids)[
                        :, 0, 0, :
                    ]
                    if logp.dtype != torch.float32:
                        logp = logp.float()
                    v, k = logp.max(1)
                    del g
                    k_is_blank = k == self._blank_index
                    blank_mask.bitwise_or_(k_is_blank)
                    del k_is_blank
                    del logp
                    blank_mask_prev.bitwise_or_(blank_mask)
                    if blank_mask.all():
                        not_blank = False
                    else:
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)
                        if hidden is not None:
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, hidden, blank_indices)
                        elif len(blank_indices) > 0 and hidden is None:
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, None, blank_indices, value=0.0)
                        k[blank_indices] = last_label[blank_indices, 0]
                        last_label = k.clone().view(-1, 1)
                        hidden = hidden_prime
                        for kidx, ki in enumerate(k):
                            if blank_mask[kidx] == 0:
                                hypotheses[kidx].y_sequence.append(ki)
                                hypotheses[kidx].timestep.append(time_idx)
                                hypotheses[kidx].score += float(v[kidx])
                        symbols_added += 1
        return hypotheses
