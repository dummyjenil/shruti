# Shruti

```python
from shruti import ShrutiASR
asr = ShrutiASR()
srt = asr("path_of_audio")
print("".join(srt.splitlines()[2::4]).replace("▁"," ").replace("<line>","\n"))
```

* SAVE TOKENIZER & MODEL

```python
import torch
from pathlib import Path
for i,v in asr.model.tokenizer_cfg['langs'].items():
  asr.model._extract_tokenizer_from_config(v,Path("langs") / i)
config = asr.model.cfg
torch.save(asr.state_dict(), "asr.pt")

```

```bash
cd langs && zip -r /content/langs.zip *
```

* SAVE TOKENIZER & MODEL

```python
from shruti.core import ShrutiASR
asr = ShrutiASR("langs") # for loading tokenizer
asr.load_model("asr.pt")
```
