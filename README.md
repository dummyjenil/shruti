# Shruti

```python
from shruti import ShrutiASR
import gradio as gr
asr = ShrutiASR()
def fn(x):
    for srt in asr(x):
        yield "".join(srt.splitlines()[2::4]).replace("▁"," ").replace("<line>","\n")
gr.Interface(fn,gr.Audio(type='filepath'),gr.TextArea()).launch()
```
