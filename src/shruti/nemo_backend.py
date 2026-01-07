import sys
from shruti import nemo
from shruti.utils import make_srt
sys.modules['nemo'] = nemo
import torchaudio
from nemo.core.classes.modelPT import ModelPT
import json
import webrtcvad
from pathlib import Path
import torch
import gc
from omegaconf import OmegaConf
from safetensors.torch import save_file
import torchaudio.functional as F_torchaudio
from huggingface_hub import hf_hub_download
import srt
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

class ShrutiASR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model_path = hf_hub_download("shethjenil/CONFORMER_INDIC_STT","indicconformer_stt_all_hybrid_rnnt_large.nemo")
        self.model = ModelPT.restore_from(model_path)
        self.eval()
        self.denormalize = self.model.cfg["preprocessor"]["window_stride"] * self.model.cfg["encoder"]["subsampling_factor"]
        # get language self.model.tokenizer.langs

    def forward(self, audio_path, lang="hi", batch_size=4,format=None,verbose=False):
        with torch.inference_mode():
            chunks, ts = make_chunks(audio_path,format=format)
            srt_ = make_srt(self.model.transcribe(chunks, language_id=lang,batch_size=batch_size, return_hypotheses=True, verbose=verbose)[0],ts,self.model.tokenizer.tokenizers_dict[lang].vocab)
            torch.cuda.empty_cache()
            gc.collect()
            return srt.compose(srt_)

    def export_to_torch(self,folder):
        folder = Path(folder)
        for i,v in self.model.tokenizer_cfg['langs'].items():
            self.model._extract_tokenizer_from_config(v,folder/"tokenizer" / i)
        json.dump(OmegaConf.to_container(self.model.to_config_dict(), resolve=True),(folder/"config.json").open("w",encoding="utf-8"),ensure_ascii=False)
        save_file(self.model.state_dict(),(folder/"model.safetensors"))
