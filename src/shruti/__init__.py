import io
from pydub import AudioSegment
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
import gc
sys.modules['nemo'] = nemo

@contextmanager
def mute_logging():
    previous_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_level)

# Note:- it is removing white noices and reduce audio size in test case 30 sec to 18 sec so we are merging a audio in forward
def make_chunks(file_path, aggressiveness=2, min_chunk_sec=10, max_chunk_sec=20, frame_ms=30):
    """
    Splits an audio file into speech chunks using WebRTC VAD.
    Everything runs purely on CPU (no GPU allocations).
    Returns:
        chunks (List[torch.Tensor]): list of float32 tensors [-1, 1]
        times (List[Tuple[float, float]]): start and end times (seconds)
    """

    # Always ensure we’re on CPU
    device = torch.device("cpu")

    # 🔹 Load audio to CPU
    wav, sr = torchaudio.load(file_path, normalize=True)
    wav = wav.to(device)

    # 🔹 Convert to mono
    wav = wav.mean(dim=0, keepdim=True)

    # 🔹 Resample to 16 kHz if needed
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    # 🔹 Convert to int16 PCM on CPU
    wav_int16 = (wav * 32768.0).clamp(-32768, 32767).short().squeeze(0)

    # 🔹 Frame segmentation for VAD
    frame_len = int(sr * frame_ms / 1000)
    total_frames = wav_int16.numel() // frame_len
    wav_int16 = wav_int16[: total_frames * frame_len]
    frames = wav_int16.view(total_frames, frame_len)

    # 🔹 Initialize WebRTC VAD (CPU)
    vad = webrtcvad.Vad(aggressiveness)
    is_speech = torch.zeros(total_frames, dtype=torch.bool)

    # 🔹 Run VAD per frame on CPU
    for i, f in enumerate(frames):
        try:
            is_speech[i] = vad.is_speech(f.numpy().tobytes(), sr)
        except Exception:
            is_speech[i] = False

    # 🔹 Build speech segments
    segs, start_idx = [], None
    for i, s in enumerate(is_speech):
        if s and start_idx is None:
            start_idx = i
        elif not s and start_idx is not None:
            segs.append((start_idx, i))
            start_idx = None
    if start_idx is not None:
        segs.append((start_idx, len(is_speech)))

    # 🔹 Merge small segments into chunks
    chunks, times = [], []
    chunk = torch.tensor([], dtype=torch.int16, device=device)
    current_time = 0.0
    chunk_start = 0.0

    for start, end in segs:
        seg = frames[start:end].flatten()
        seg_len_sec = len(seg) / sr
        chunk_len_sec = len(chunk) / sr

        if chunk_len_sec + seg_len_sec <= max_chunk_sec:
            if len(chunk) == 0:
                chunk_start = current_time
            chunk = torch.cat([chunk, seg])
        else:
            if chunk_len_sec >= min_chunk_sec:
                end_s = chunk_start + chunk_len_sec
                chunks.append(chunk.clone())
                times.append((round(chunk_start, 2), round(end_s, 2)))
                current_time = end_s
                chunk = seg
                chunk_start = current_time
            else:
                chunk = torch.cat([chunk, seg])

    # 🔹 Add last chunk
    if len(chunk) > 0:
        end_s = chunk_start + len(chunk) / sr
        chunks.append(chunk)
        times.append((round(chunk_start, 2), round(end_s, 2)))

    # 🔹 Convert to float [-1, 1] on CPU
    chunks = [c.to(torch.float32) / 32768.0 for c in chunks]

    return chunks, times

def generate_audio(chunks: list[torch.Tensor], bitrate="64k"):
    """
    Concatenate multiple audio chunks (float tensors) and export as OGG.
    Works entirely on CPU — safe even if chunks are on GPU.
    """
    ogg_bytes = io.BytesIO()

    # ✅ Move all chunks to CPU early to avoid GPU memory usage
    with torch.no_grad():
        chunks_cpu = [c.detach().to("cpu", torch.float32) for c in chunks]

        # Concatenate on CPU
        audio = torch.cat(chunks_cpu, dim=-1)

        # Clamp and convert to int16 PCM (CPU only)
        audio = torch.clamp(audio, -1.0, 1.0)
        audio = (audio * 32767).short()

        # Convert to bytes and export to OGG
        AudioSegment(
            data=audio.numpy().tobytes(),
            sample_width=2,   # 16-bit PCM
            frame_rate=16000, # Hz
            channels=1
        ).export(ogg_bytes, format="ogg", bitrate=bitrate)

    return ogg_bytes.getvalue()

class ShrutiASR(torch.nn.Module):

    def __init__(self, model_path=None):
        super().__init__()
        if not model_path:
            model_path = hf_hub_download("shethjenil/CONFORMER_INDIC_STT","indicconformer_stt_all_hybrid_rnnt_large.nemo")
        with mute_logging():
            self.model:EncDecHybridRNNTCTCBPEModel = ASRModel.restore_from(model_path)
        self.model.eval()
        self.denormalize = self.model.to_config_dict()['preprocessor']['window_stride'] * self.model.encoder.subsampling_factor
        self.language = list(self.model.tokenizer.tokenizers_dict.keys())

    def forward(self,audio_path,lang="gu",batch_size=4,get_audio=True):
        chunks , ts = make_chunks(audio_path)

        # with torch.no_grad():
        #     hyp:list[Hypothesis] = self.model.transcribe(chunks, language_id=lang,batch_size=batch_size,return_hypotheses=True,verbose=False)[0]

        all_hyp = []
        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                sub = chunks[i:i+batch_size]
                out = self.model.transcribe(
                    sub, language_id=lang, batch_size=len(sub),
                    return_hypotheses=True, verbose=False
                )[0]
                all_hyp.extend(out)
                torch.cuda.empty_cache()
        hyp = all_hyp



        vocab:SentencePieceTokenizer = self.model.tokenizer.tokenizers_dict[lang].vocab
        timestamp = []
        for h, (s, e) in zip(hyp, ts):
            starts = s + np.array(h.timestep) * self.denormalize
            for txt, st, en in zip([vocab[y] for y in h.y_sequence.tolist()], starts, list(starts[1:]) + [e]):
                timestamp.append({"text": txt, "start": float(st), "end": float(en)})
            timestamp.append({"text": "<line>", "start": float(e), "end": float(e + 0.005)})
        del hyp
        gc.collect()
        torch.cuda.empty_cache()
        output = generate_audio(chunks) if get_audio else None
        del chunks
        gc.collect()
        torch.cuda.empty_cache()
        return srt.compose([
        srt.Subtitle(
            index,
            timedelta(seconds=i["start"]),
            timedelta(seconds=i["end"]),
            i["text"]
        )
        for index, i in enumerate(timestamp, 1)
    ]),output
