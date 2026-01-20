from shruti.utils import ChunkedData, make_srt
import torchaudio
import torch
import gc
import srt
from huggingface_hub import hf_hub_download
import sys
import os
import logging
import warnings
from contextlib import contextmanager

@contextmanager
def absolute_silence():
    # save states
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    saved_disable = logging.root.manager.disable
    saved_filters = warnings.filters[:]

    # kill everything
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    try:
        yield
    finally:
        # restore everything
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        logging.disable(saved_disable)
        warnings.filters = saved_filters

class ShrutiASR(torch.nn.Module):
    @absolute_silence()
    def __init__(self,language='all'):
        super().__init__()
        from shruti import nemo
        sys.modules['nemo'] = nemo
        from nemo.core.classes.modelPT import ModelPT
        
        self.model = ModelPT.restore_from(hf_hub_download("shethjenil/CONFORMER_INDIC_STT",f"indicconformer_stt_{language}_hybrid_rnnt_large.nemo"))
        self.denormalizer = self.model.cfg["preprocessor"]["window_stride"] * self.model.cfg["encoder"]["subsampling_factor"]
        self.eval()

    @torch.inference_mode()
    def forward(self, audio_path, batch_size=2, language="hi",use_tqdm=False):
        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, sr, self.preprocessor.sr)
        ds = ChunkedData(wav,16000)
        hyp = self.model.transcribe(ds.data, language_id=language,batch_size=batch_size, return_hypotheses=True, verbose=use_tqdm)[0]
        torch.cuda.empty_cache()
        gc.collect()
        return srt.compose(make_srt(hyp, ds.ts, self.model.tokenizer.tokenizers_dict[language].vocab,self.denormalizer))

    def export(self,folder):
        import json
        from pathlib import Path
        from omegaconf import OmegaConf
        from safetensors.torch import save_file

        folder = Path(folder)
        folder.mkdir(True,True)
        for i,v in self.model.tokenizer_cfg['langs'].items():
            self.model._extract_tokenizer_from_config(v,folder/"tokenizer" / i)
        json.dump(OmegaConf.to_container(self.model.to_config_dict(), resolve=True),(folder/"config.json").open("w",encoding="utf-8"),ensure_ascii=False)
        save_file(self.model.state_dict(),(folder/"model.safetensors"))
