# Shruti

```python
from shruti import ShrutiASR
asr = ShrutiASR()
srt = asr("path_of_audio")
print("".join(srt.splitlines()[2::4]).replace("▁"," ").replace("<line>","\n"))
```
