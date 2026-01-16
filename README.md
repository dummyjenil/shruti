# Shruti

```python
from shruti import ShrutiASR
import gradio as gr
asr = ShrutiASR()
def fn(x):
    for srt in asr(x):
        yield "".join(srt.splitlines()[2::4]).replace("▁"," ").replace("<line>","\n")
gr.Interface(fn,gr.Audio(),gr.TextArea()).launch()
```
<!-- ```python
import shruti.nemo_backend
import safetensors.torch
safetensors.torch.save_file(shruti.nemo_backend.ModelPT.from_pretrained("ai4bharat/IndicConformer").state_dict(),"model.safetensors")
``` -->
