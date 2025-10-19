# Shruti

```python
import gradio as gr
from shruti import ShrutiASR
asr = ShrutiASR()
gr.Interface(asr,[gr.Audio(type="filepath"),gr.Dropdown(['sentence', 'char', 'word'],value="word"),gr.Dropdown(asr.language),gr.Slider(1,16,4,step=1)],gr.TextArea()).launch()
```
