# Shruti

```python
from shruti import ShrutiASR
asr = ShrutiASR()
srt = asr("path_of_audio")
print("".join(srt.splitlines()[2::4]).replace("▁"," ").replace("<line>","\n"))
```

<!-- ```python
import shruti.nemo_backend
import safetensors.torch
safetensors.torch.save_file(shruti.nemo_backend.ModelPT.from_pretrained("ai4bharat/IndicConformer").state_dict(),"model.safetensors")
``` -->
