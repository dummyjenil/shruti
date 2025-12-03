import sys
from shruti import nemo
sys.modules['nemo'] = nemo
import json
from pathlib import Path
from nemo.core.classes.modelPT import ModelPT
from shruti.utils import make_chunks, make_srt
import torch
import gc
from omegaconf import OmegaConf
from safetensors.torch import save_file

def edit_state_dict(state_dict,src,replace):
  new_state_dict = {}
  for k,v in state_dict.items():
    if k.startswith(src):
      new_state_dict[k.replace(src,replace,1)] = v
    else:
      new_state_dict[k] = v
  return new_state_dict

class ShrutiASR(torch.nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.model = ModelPT.restore_from(model_path)
        self.eval()
        self.denormalize = self.model.cfg["preprocessor"]["window_stride"] * self.model.cfg["encoder"]["subsampling_factor"]
        # get language self.model.tokenizer.langs

    def forward(self, audio_path, lang="hi", batch_size=4,format=None,verbose=False):
        with torch.inference_mode():
            chunks, ts = make_chunks(audio_path,format=format)
            srt = make_srt(self.model.transcribe(chunks, language_id=lang,batch_size=batch_size, return_hypotheses=True, verbose=verbose)[0],ts,self.denormalize,self.model.tokenizer.tokenizers_dict[lang].vocab)
            torch.cuda.empty_cache()
            gc.collect()
            return srt

    def export_to_torch(self,folder):
        folder = Path(folder)
        for i,v in self.model.tokenizer_cfg['langs'].items():
            self.model._extract_tokenizer_from_config(v,folder/"tokenizer" / i)
        json.dump(OmegaConf.to_container(self.model.to_config_dict(), resolve=True),(folder/"config.json").open("w",encoding="utf-8"),ensure_ascii=False)
        m = self.model.state_dict()
        m = edit_state_dict(m,"encoder.pre_encode","pre_encode")
        m = edit_state_dict(m,"encoder.layers","enc_layers")
        save_file(m,(folder/"model.safetensors"))
