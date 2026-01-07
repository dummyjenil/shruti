from tqdm import tqdm
from typing import Dict, Tuple
from torch import nn, Tensor
from torch.nn import functional as F
import torch
import torchaudio
import math
import gc
def batchify(tensor: Tensor, T: int) -> Tensor:
    orig_size = tensor.size(-1)
    new_size = math.ceil(orig_size / T) * T
    tensor = F.pad(tensor, [0, new_size - orig_size])
    return torch.cat(torch.split(tensor, T, dim=-1), dim=0)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 5, 2)
        self.bn = nn.BatchNorm2d(out_channels,0.001,0.01)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        down = self.conv(F.pad(input, (1, 2, 1, 2), "constant", 0))
        return down, self.relu(self.bn(down))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels,5,2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels,0.001, 0.01)

    def forward(self, input: Tensor) -> Tensor:
        up = self.tconv(input)
        # reverse padding
        l, r, t, b = 1, 2, 1, 2
        up = up[:, :, l:-r, t:-b]
        return self.bn(self.relu(up))

class UNet(nn.Module):
    def __init__(
        self,
        n_layers: int = 6,
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        down_set = [in_channels] + [2 ** (i + 4) for i in range(n_layers)]
        self.encoder_layers = nn.ModuleList([EncoderBlock(in_ch, out_ch) for in_ch, out_ch in zip(down_set[:-1], down_set[1:])])
        up_set = [1] + [2 ** (i + 4) for i in range(n_layers)]
        up_set.reverse()
        self.decoder_layers = nn.ModuleList([DecoderBlock(in_ch if i == 0 else in_ch * 2,out_ch) for i, (in_ch, out_ch) in enumerate(zip(up_set[:-1], up_set[1:]))])
        self.up_final = nn.Conv2d(1, in_channels,4,1,3,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        encoder_outputs_pre_act = []
        x = input
        for down in self.encoder_layers:
            conv, x = down(x)
            encoder_outputs_pre_act.append(conv)

        for i, up in enumerate(self.decoder_layers):
            if i == 0:
                x = up(encoder_outputs_pre_act.pop())
            else:
                # merge skip connection
                x = up(torch.cat([encoder_outputs_pre_act.pop(), x], dim=1))

        mask = self.sigmoid(self.up_final(x))

        # --- Crop both mask and input to match in size ---
        min_f = min(mask.size(-2), input.size(-2))
        min_t = min(mask.size(-1), input.size(-1))
        mask = mask[..., :min_f, :min_t]
        input = input[..., :min_f, :min_t]
        # -------------------------------------------------

        return mask * input

class Spleeter(nn.Module):

    def __init__(self, instrument_models):
        super().__init__()
        self.F = 1024
        self.T = 512
        self.win_length = 4096
        self.hop_length = 1024
        self.win = torch.hann_window(self.win_length)
        self.stems = nn.ModuleDict({name: UNet() for name in instrument_models})
        self.eval()

    def compute_stft(self, wav: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes STFT feature from wav
        Args:
            wav (Tensor): B x L or 2 x L for stereo
        Returns:
            stft (Tensor): B x F x T x 2 (real+imag)
            mag (Tensor): B x F x T magnitude
        """
        stft = torch.stft(
            wav,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            window=self.win,
            center=True,
            return_complex=False,  # keep old format
            pad_mode="constant",
        )

        # Keep only first F frequency bins
        stft = stft[:, :self.F, :, :]

        # magnitude
        real = stft[:, :, :, 0]
        imag = stft[:, :, :, 1]
        mag = torch.sqrt(real**2 + imag**2 + 1e-10)

        return stft, mag

    def inverse_stft(self, stft: Tensor) -> Tensor:
        """Inverse STFT from real+imag tensor (B x F x T x 2)"""

        # Ensure frequency dimension matches n_fft / 2 + 1
        target_F = self.win_length // 2 + 1
        if stft.size(1) < target_F:
            pad = target_F - stft.size(1)
            stft = F.pad(stft, (0, 0, 0, 0, 0, pad))  # pad along freq dim

        # Convert real+imag to complex for istft
        stft_complex = torch.view_as_complex(stft)

        wav = torch.istft(
            stft_complex,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.win,
        )

        return wav.detach()

    def forward(self, wav: Tensor, batch_size=16, device="cpu") -> Dict[str, Tensor]:
        stft, stft_mag = self.compute_stft(wav)
        L = stft.size(2)

        stft_mag = stft_mag.unsqueeze(-1).permute([3, 0, 1, 2])  # B x 2 x F x T
        stft_mag = batchify(stft_mag, self.T).transpose(2, 3)    # B x 2 x T x F

        # GPU-safe batch inference
        masks = self.infer_with_batches(stft_mag, batch_size, device)

        mask_sum = sum([m**2 for m in masks.values() if m.numel() > 0])
        mask_sum += 1e-10

        def apply_mask(mask):
            mask = (mask**2 + 1e-10 / 2) / mask_sum
            mask = mask.transpose(2, 3)
            mask = torch.cat(torch.split(mask, 1, dim=0), dim=3)
            mask = mask.squeeze(0)[:, :, :L].unsqueeze(-1)
            stft_masked = stft * mask  # dono GPU pe honge
            return stft_masked

        return {name: self.inverse_stft(apply_mask(m)) for name, m in masks.items() if m.numel() > 0}

    def infer_with_batches(self, stft_mag, batch_size, device):
        masks = {name: [] for name in self.stems.keys()}

        for i in range(0, stft_mag.shape[0], batch_size):
            batch = stft_mag[i:i + batch_size].to(device)
            batch_outputs = {name: net(batch) for name, net in self.stems.items()}

            for name in batch_outputs:
                masks[name].append(batch_outputs[name])  # GPU pe hi rakho

            del batch, batch_outputs
            torch.cuda.empty_cache()
            gc.collect()

        final_masks = {}
        for name, lst in masks.items():
            if lst:
                final_masks[name] = torch.cat(lst, dim=0)
            else:
                final_masks[name] = torch.empty(0, device=device)

        return final_masks

    torch.inference_mode()
    def separate_audio_in_chunks(
        self,
        wav, 
        sr,
        chunk_duration_sec=30,
        batch_size=2,
    ):
        """
        Memory-safe audio separation
        """
        device = next(self.parameters()).device
        target_sr = 44100
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        if wav.dim() == 2 and wav.size(0) == 1:
            wav = wav.repeat(2, 1)
        wav = wav.to(device)
        samples_per_chunk = chunk_duration_sec * target_sr

        separated_outputs = {name: [] for name in self.stems.keys()}

        total_samples = wav.shape[-1]

        for start in tqdm(range(0, total_samples, samples_per_chunk)):
            end = min(start + samples_per_chunk, total_samples)
            wav_chunk = wav[:, start:end]

            chunk_result = self.forward(
                wav_chunk,
                batch_size=batch_size,
                device=device
            )

            # 🔥 IMPORTANT: move outputs to CPU immediately
            for name, audio in chunk_result.items():
                separated_outputs[name].append(audio.cpu())

            del chunk_result, wav_chunk
            torch.cuda.empty_cache()
            gc.collect()

        # concatenate all chunks
        final_outputs = {
            name: torch.cat(chunks, dim=-1)
            for name, chunks in separated_outputs.items()
        }

        return final_outputs
