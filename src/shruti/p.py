import torch
from torchaudio.models import emformer_rnnt_model

model = emformer_rnnt_base(num_symbols=vocab_size)

@torch.no_grad()
def greedy_decode_rnnt(model, enc, enc_len, blank):
    B, T, _ = enc.shape
    state = None
    y = torch.full((B, 1), blank, device=enc.device, dtype=torch.long)
    hyps = [[] for _ in range(B)]

    for t in range(T):
        f = enc[:, t:t+1]
        symbols = 0

        while True:
            g, _, state = model.predict(y, torch.ones(B, device=y.device), state)
            logits, _, _ = model.join(f, torch.ones(B), g, torch.ones(B))
            k = logits[:, 0, 0].argmax(-1)

            if (k == blank).all():
                break

            for b in range(B):
                if k[b] != blank:
                    hyps[b].append(int(k[b]))

            y = k.view(B, 1)
            symbols += 1
            if symbols > 10:
                break

    return hyps
