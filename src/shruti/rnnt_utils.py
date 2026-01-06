import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

@dataclass
class Hypothesis:
    score: float
    y_sequence: Union[List[int], torch.Tensor]
    timestep: Union[List[int], torch.Tensor] = field(default_factory=list)

def label_collate(labels):
    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")
    batch_size = len(labels)
    max_len = max(len(label) for label in labels)
    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        if isinstance(l, torch.Tensor):
            l = l.squeeze().cpu().numpy()
            if l.ndim == 0:
                l = [l.item()]
        cat_labels[e, : len(l)] = l
    return torch.tensor(cat_labels, dtype=torch.int64)

class RNNTDecoder(nn.Module):
    def __init__(
        self,
        pred_hidden:int,
        pred_rnn_layers:int,
        vocab_size: int,
    ):
        super().__init__()
        rnn_hidden_size = -1
        self.pred_hidden = pred_hidden
        self.embed = nn.Embedding(vocab_size + 1, self.pred_hidden, padding_idx=vocab_size)
        self.lstm = nn.LSTM(self.pred_hidden,rnn_hidden_size if rnn_hidden_size > 0 else self.pred_hidden,pred_rnn_layers)
    def forward(self, targets, target_length, states=None):
        g, states = self.predict(label_collate(targets), state=states)
        return g.transpose(1, 2), target_length, states
    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype
        if y is not None:
            if y.device != device:
                y = y.to(device)
            y = self.embed(y)
        else:
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size
            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)
        B, U, H = y.shape
        start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
        y = torch.cat([start, y], dim=1).contiguous()  
        y = y.transpose(0, 1)  
        g, hid = self.lstm(y, state)
        g = g.transpose(0, 1)  
        del y, start, state
        return g, hid

    def batch_copy_states(
        self,
        old_states: List[torch.Tensor],
        new_states: List[torch.Tensor],
        ids: List[int],
        value: Optional[float] = None,
    ) -> List[torch.Tensor]:
        for state_id in range(len(old_states)):
            if value is None:
                old_states[state_id][:, ids, :] = new_states[state_id][:, ids, :]
            else:
                old_states[state_id][:, ids, :] *= 0.0
                old_states[state_id][:, ids, :] += value
        return old_states

class RNNTJoint(nn.Module):
    def __init__(
        self,
        pred_hidden:int,
        encoder_hidden:int,
        joint_hidden:int,
        num_classes: int,
        vocabulary: Optional[List] = None,
        multilingual: bool = False, 
        language_keys: Optional[List] = None, 
        offset_token_ids_by_token_id=None, 
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self._vocab_size = num_classes
        self._num_classes = num_classes + 1  
        self.offset_token_ids_by_token_id = offset_token_ids_by_token_id 
        self.multilingual = multilingual 
        self.language_keys = language_keys 
        self.pred = nn.Linear(pred_hidden, joint_hidden)
        self.enc = nn.Linear(encoder_hidden, joint_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.joint_net = nn.ModuleDict({lang:nn.Linear(joint_hidden, (self._vocab_size // len(self.language_keys)+1)) for lang in self.language_keys}) if self.multilingual else nn.Linear(joint_hidden, self._num_classes)
        self.temperature = 1.0
    def forward(self,encoder_outputs: torch.Tensor,decoder_outputs: Optional[torch.Tensor],joint_net:nn.Module) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        encoder_outputs = encoder_outputs.transpose(1, 2)
        decoder_outputs = decoder_outputs.transpose(1, 2)
        f = self.enc(encoder_outputs).unsqueeze(dim=2)
        g = self.pred(decoder_outputs).unsqueeze(dim=1)
        x = f + g
        del f, g
        x = self.relu(x)
        x = joint_net(x)
        if not x.is_cuda:  
            if self.temperature != 1.0:
                x = (x / self.temperature).log_softmax(dim=-1)
            else:
                x = x.log_softmax(dim=-1)
        return x


class GreedyRNNTInfer(nn.Module):
    def __init__(self,pred_hidden,pred_rnn_layers,vocab_size,d_model,joint_hidden,blank_id,max_symbols,language_keys):
        super().__init__()
        self.decoder = RNNTDecoder(pred_hidden,pred_rnn_layers,vocab_size)
        self.joint = RNNTJoint(pred_hidden,d_model,joint_hidden,vocab_size,multilingual=True,language_keys=language_keys)
        self._blank_index = blank_id
        self._SOS = blank_id
        self.max_symbols = max_symbols

    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        hidden: Optional[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(label, torch.Tensor):
            if label.dtype != torch.long:
                label = label.long()
            return self.decoder.predict(label, hidden, batch_size=batch_size)
        else:
            if label == self._SOS:
                return self.decoder.predict(None, hidden, batch_size=batch_size)
            label = label_collate([[label]])
            return self.decoder.predict(label, hidden, batch_size=batch_size)

    def _joint_step(self, enc, pred, lid):
        logits = self.joint(enc.transpose(1, 2), pred.transpose(1, 2), self.joint.joint_net[lid] if self.joint.multilingual else self.joint.joint_net)
        if not logits.is_cuda:
            logits = logits.log_softmax(dim=len(logits.shape) - 1)
        return logits

    torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        lid=None,
    ):
        batchsize = x.shape[0]
        hypotheses = [Hypothesis(score=0.0, y_sequence=[], timestep=[]) for _ in range(batchsize)]
        hidden = None
        last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long, device=device)
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)
        blank_mask_prev = None
        max_out_len = out_len.max()
        for time_idx in range(max_out_len):
            f = x.narrow(dim=1, start=time_idx, length=1)  
            not_blank = True
            symbols_added = 0
            blank_mask.mul_(False)
            blank_mask = time_idx >= out_len
            blank_mask_prev = blank_mask.clone()
            while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                if time_idx == 0 and symbols_added == 0 and hidden is None:
                    g, hidden_prime = self._pred_step(self._SOS, hidden, batch_size=batchsize)
                else:
                    g, hidden_prime = self._pred_step(last_label, hidden, batch_size=batchsize)
                logp = self._joint_step(f, g,lid)[
                    :, 0, 0, :
                ]
                if logp.dtype != torch.float32:
                    logp = logp.float()
                v, k = logp.max(1)
                del g
                k_is_blank = k == self._blank_index
                blank_mask.bitwise_or_(k_is_blank)
                del k_is_blank
                del logp
                blank_mask_prev.bitwise_or_(blank_mask)
                if blank_mask.all():
                    not_blank = False
                else:
                    blank_indices = (blank_mask == 1).nonzero(as_tuple=False)
                    if hidden is not None:
                        hidden_prime = self.decoder.batch_copy_states(hidden_prime, hidden, blank_indices)
                    elif len(blank_indices) > 0 and hidden is None:
                        hidden_prime = self.decoder.batch_copy_states(hidden_prime, None, blank_indices, value=0.0)
                    k[blank_indices] = last_label[blank_indices, 0]
                    last_label = k.clone().view(-1, 1)
                    hidden = hidden_prime
                    for kidx, ki in enumerate(k):
                        if blank_mask[kidx] == 0:
                            hypotheses[kidx].y_sequence.append(ki)
                            hypotheses[kidx].timestep.append(time_idx)
                            hypotheses[kidx].score += float(v[kidx])
                    symbols_added += 1
        return hypotheses
