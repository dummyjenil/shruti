import sys
from shruti import nemo

sys.modules['nemo'] = nemo

from nemo.collections.asr.models import ASRModel
from contextlib import contextmanager
from datetime import timedelta
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
import numpy as np
import torch
import torchaudio
import webrtcvad
import srt
import logging
import gc

@contextmanager
def mute_logging():
    previous_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_level)

def make_chunks(file_path,aggressiveness=2,min_chunk_sec=10,max_chunk_sec=15,frame_ms=30,format=None):
    wav, sr = torchaudio.load(file_path, normalize=True,format=format)
    wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
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
    chunks, times = [], []
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
        times.append((round(start_sample / sr, 2), round(end_sample / sr, 2)))
        start_sample = end_sample
    return chunks, times


class ShrutiASR(torch.nn.Module):

    def __init__(self, model_path=None):
        super().__init__()
        if not model_path:
            model_path = hf_hub_download("shethjenil/CONFORMER_INDIC_STT","indicconformer_stt_all_hybrid_rnnt_large.nemo")
        with mute_logging():
            self.model:EncDecHybridRNNTCTCBPEModel = ASRModel.restore_from(model_path)
        self.eval()
        self.denormalize = self.model.to_config_dict()["preprocessor"]["window_stride"] * self.model.encoder.subsampling_factor
        self.languages = list(self.model.tokenizer.tokenizers_dict.keys())

    def forward(self, audio_path, lang="gu", batch_size=4,format=None,verbose=False):
        vocab: SentencePieceTokenizer = self.model.tokenizer.tokenizers_dict[lang].vocab
        with torch.inference_mode():
            chunks, ts = make_chunks(audio_path,format=format)
            timestamp = []
            for h, (s, e) in zip(self.model.transcribe(chunks, language_id=lang,batch_size=batch_size, return_hypotheses=True, verbose=verbose)[0], ts):
                starts = s + np.array(h.timestep) * self.denormalize
                for txt, st, en in zip([vocab[y] for y in h.y_sequence.tolist()],starts,list(starts[1:]) + [e]):
                    timestamp.append({"text": txt,"start": float(st),"end": float(en)})
                timestamp.append({"text": "<line>","start": float(e),"end": float(e + 0.005)})
            torch.cuda.empty_cache()
            gc.collect()
            return srt.compose([srt.Subtitle(index,timedelta(seconds=i["start"]),timedelta(seconds=i["end"]),i["text"]) for index, i in enumerate(timestamp, 1)])
