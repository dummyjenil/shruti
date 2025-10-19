# Shruti

```python
import gradio as gr
from shruti import ShrutiASR , generate_srt
asr = ShrutiASR()
def func(input_audio,type_of_transcribe,lang,batch_size,progress=gr.Progress(track_tqdm=True)):
    open("sub.srt","w").write(generate_srt(asr.forward(input_audio,type_of_transcribe,lang,batch_size)))
    return "sub.srt"
gr.Interface(func,[gr.Audio(type="filepath"),gr.Dropdown(['sentence', 'char', 'word'],value="word"),gr.Dropdown(asr.language),gr.Slider(1,16,4,step=1)],gr.File()).launch()
```
