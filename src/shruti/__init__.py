from typing import Literal
from shruti.nemo.collections.asr.models import ASRModel
from contextlib import contextmanager
from shruti import nemo
from datetime import timedelta
from huggingface_hub import hf_hub_download
from shruti.nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from shruti.nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from shruti.nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
import numpy as np
import torch
import torchaudio
import webrtcvad
import srt
import logging
import sys
sys.modules['nemo'] = nemo

@contextmanager
def mute_logging():
    previous_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_level)

def make_chunks(file_path, aggressiveness=2, min_chunk_sec=10, max_chunk_sec=20, frame_ms=30):
    wav, sr = torchaudio.load(file_path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    wav_int16 = (wav * 32768.0).clamp(-32768, 32767).short().squeeze(0)
    frame_len = int(sr * frame_ms / 1000)
    total_frames = wav_int16.numel() // frame_len
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
    chunk = torch.tensor([], dtype=torch.int16)
    chunk_start = 0.0

    for start, end in segs:
        seg = frames[start:end].flatten()
        seg_start = start * frame_ms / 1000.0
        seg_len_ms = len(seg) * 1000 / sr
        chunk_len_ms = len(chunk) * 1000 / sr

        if chunk_len_ms + seg_len_ms <= max_chunk_sec * 1000:
            if len(chunk) == 0:
                chunk_start = seg_start
            chunk = torch.cat([chunk, seg])
        else:
            if chunk_len_ms >= min_chunk_sec * 1000:
                end_s = chunk_start + len(chunk) / sr
                chunks.append(chunk.clone())
                times.append((chunk_start, end_s))
                chunk = seg
                chunk_start = seg_start
            else:
                chunk = torch.cat([chunk, seg])

    if len(chunk) > 0:
        end_s = chunk_start + len(chunk) / sr
        chunks.append(chunk)
        times.append((chunk_start, end_s))

    return [c.float() / 32768.0 for c in chunks], times

class ShrutiASR(torch.nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        if not model_path:
            model_path = hf_hub_download("shethjenil/CONFORMER_INDIC_STT","indicconformer_stt_all_hybrid_rnnt_large.nemo")
        with mute_logging():
            self.model:EncDecHybridRNNTCTCBPEModel = ASRModel.restore_from(model_path)
        self.model.eval()
        self.model.cur_decoder = "rnnt"
        self.denormalize = self.model.to_config_dict()['preprocessor']['window_stride'] * self.model.encoder.subsampling_factor
        self.language = list(self.model.tokenizer.tokenizers_dict.keys())
    def forward(self,audio_path,type_of_transcribe:Literal['sentence','char','word']='word',lang="gu",batch_size=4):
        chunks , ts = make_chunks(audio_path)
        hyp:list[Hypothesis] = self.model.transcribe(chunks, language_id=lang,batch_size=batch_size,return_hypotheses=True)[0]
        vocab:SentencePieceTokenizer = self.model.tokenizer.tokenizers_dict.get(lang).vocab
        if type_of_transcribe == "sentence":
            return [{"text":i.text,"start":s,"end":e} for i,(s,e) in zip(hyp,ts)]
        elif type_of_transcribe == "char":
            srt = []
            for h, (s, e) in zip(hyp, ts):
                starts = s + np.array(h.timestep) * self.denormalize
                texts = [vocab[int(y)] for y in h.y_sequence]

                # Compute end times — next start or the chunk end
                ends = list(starts[1:]) + [e]

                for txt, st, en in zip(texts, starts, ends):
                    srt.append({"text": txt, "start": float(st), "end": float(en)})

                # Optional newline marker at the end of the chunk
                srt.append({"text": "\n", "start": float(e), "end": float(e)})
            return srt
        elif type_of_transcribe == "word":
            srt = []
            for h, (s, e) in zip(hyp, ts):
                starts = s + np.array(h.timestep) * self.denormalize
                texts = [vocab[int(y)] for y in h.y_sequence]

                word, word_start = "", None
                for txt, st in zip(texts, starts):
                    if txt.startswith("▁"):  # start of new word
                        if word:  # save previous word
                            srt.append({"text": word.strip(), "start": float(word_start), "end": float(st)})
                        word = txt.replace("▁", "")  # start new word
                        word_start = st
                    else:
                        word += txt  # continue word

                # last word in chunk
                if word:
                    srt.append({"text": word.strip(), "start": float(word_start), "end": float(e)})
                srt.append({"text": "\n", "start": float(e), "end": float(e)})
            return srt


def generate_srt(srt_list):
    return srt.compose([srt.Subtitle(index,timedelta(seconds=i['start']),timedelta(seconds=i['end']),i['text']) for index,i in enumerate(srt_list,1)])
