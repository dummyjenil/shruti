import logging
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import librosa
import torch.nn.functional as F
import torchaudio
import webrtcvad
from datetime import timedelta
import srt
from torch.utils.data import Dataset
import torchaudio.functional as F_torchaudio
import math
BLANK_ID = 256
CONSTANT = 1e-5

def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for _ in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)

def label_collate(labels):
    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")
    batch_size = len(labels)
    max_len = max(len(label) for label in labels)
    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    return torch.tensor(cat_labels, dtype=torch.int64)

# @dataclass
# class Hypothesis:
#     score: float
#     y_sequence: Union[List[int], torch.Tensor]
#     text: Optional[str] = None
#     dec_out: Optional[List[torch.Tensor]] = None
#     dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
#     timestep: Union[List[int], torch.Tensor] = field(default_factory=list)
#     alignments: Optional[Union[List[int], List[List[int]]]] = None
#     frame_confidence: Optional[Union[List[float], List[List[float]]]] = None
#     token_confidence: Optional[List[float]] = None
#     word_confidence: Optional[List[float]] = None
#     length: Union[int, torch.Tensor] = 0
#     y: List[torch.tensor] = None
#     lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
#     lm_scores: Optional[torch.Tensor] = None
#     ngram_lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
#     tokens: Optional[Union[List[int], torch.Tensor]] = None
#     last_token: Optional[torch.Tensor] = None

@dataclass
class Hypothesis:
    score: float
    y_sequence: List[int]
    timestep: List[int]
    dec_state: Optional[tuple] = None


# class RNNTDecoder(nn.Module):
#     def __init__(
#         self,
#         prednet: Dict[str, Any],
#         vocab_size: int,
#         random_state_sampling: bool = False,
#         blank_as_pad: bool = True,
#         multisoftmax=False, #CTEMO
#         language_masks=None, #CTEMO
        
#     ):
#         # Required arguments
#         self.pred_hidden = prednet['pred_hidden']
#         self.pred_rnn_layers = prednet["pred_rnn_layers"]
#         self.blank_idx = vocab_size
#         self.blank_as_pad = blank_as_pad
#         # Initialize the model (blank token increases vocab size by 1)
#         super().__init__()

#         self.random_state_sampling = random_state_sampling
#         rnn_hidden_size = prednet.get("rnn_hidden_size", -1)

#         self.prediction = nn.ModuleDict(
#                     {
#                         "embed": nn.Embedding(vocab_size + 1, self.pred_hidden, padding_idx=self.blank_idx),
#                         "dec_rnn": nn.LSTM(self.pred_hidden,rnn_hidden_size if rnn_hidden_size > 0 else self.pred_hidden,self.pred_rnn_layers,proj_size=self.pred_hidden if self.pred_hidden < rnn_hidden_size else 0,),
#                     })

#         self._rnnt_export = False

#         self.multisoftmax = multisoftmax #CTEMO
#         self.language_masks = language_masks #CTEMO

#     def forward(self, targets, target_length, states=None):
#         # y: (B, U)
#         y = label_collate(targets)

#         # state maintenance is unnecessary during training forward call
#         # to get state, use .predict() method.
#         if self._rnnt_export:
#             add_sos = False
#         else:
#             add_sos = True

#         g, states = self.predict(y, state=states, add_sos=add_sos)  # (B, U, D)
#         g = g.transpose(1, 2)  # (B, D, U)

#         return g, target_length, states

#     def predict(
#         self,
#         y: Optional[torch.Tensor] = None,
#         state: Optional[List[torch.Tensor]] = None,
#         add_sos: bool = True,
#         batch_size: Optional[int] = None,
#     ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
#         _p = next(self.parameters())
#         device = _p.device
#         dtype = _p.dtype
#         if y is not None:
#             if y.device != device:
#                 y = y.to(device)
#             y = self.prediction["embed"](y)
#         else:
#             if batch_size is None:
#                 B = 1 if state is None else state[0].size(1)
#             else:
#                 B = batch_size

#             y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

#         # Prepend blank "start of sequence" symbol (zero tensor)
#         if add_sos:
#             B, U, H = y.shape
#             start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
#             y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
#         else:
#             start = None  # makes del call later easier

#         # If in training mode, and random_state_sampling is set,
#         # initialize state to random normal distribution tensor.
#         if state is None:
#             if self.random_state_sampling and self.training:
#                 state = self.initialize_state(y)

#         # Forward step through RNN
#         y = y.transpose(0, 1)  # (U + 1, B, H)
#         g, hid = self.prediction["dec_rnn"](y, state)
#         g = g.transpose(0, 1)  # (B, U + 1, H)

#         del y, start, state

#         # Adapter module forward step
#         # if self.is_adapter_available():
#         #     g = self.forward_enabled_adapters(g)

#         return g, hid

#     def initialize_state(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Initialize the state of the LSTM layers, with same dtype and device as input `y`.
#         LSTM accepts a tuple of 2 tensors as a state.

#         Args:
#             y: A torch.Tensor whose device the generated states will be placed on.

#         Returns:
#             Tuple of 2 tensors, each of shape [L, B, H], where

#                 L = Number of RNN layers

#                 B = Batch size

#                 H = Hidden size of RNN.
#         """
#         batch = y.size(0)
#         if self.random_state_sampling and self.training:
#             state = (
#                 torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
#                 torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
#             )

#         else:
#             state = (
#                 torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
#                 torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
#             )
#         return state

#     def score_hypothesis(
#         self, hypothesis: Hypothesis, cache: Dict[Tuple[int], Any]
#     ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
#         """
#         Similar to the predict() method, instead this method scores a Hypothesis during beam search.
#         Hypothesis is a dataclass representing one hypothesis in a Beam Search.

#         Args:
#             hypothesis: Refer to Hypothesis.
#             cache: Dict which contains a cache to avoid duplicate computations.

#         Returns:
#             Returns a tuple (y, states, lm_token) such that:
#             y is a torch.Tensor of shape [1, 1, H] representing the score of the last token in the Hypothesis.
#             state is a list of RNN states, each of shape [L, 1, H].
#             lm_token is the final integer token of the hypothesis.
#         """
#         if hypothesis.dec_state is not None:
#             device = hypothesis.dec_state[0].device
#         else:
#             _p = next(self.parameters())
#             device = _p.device

#         # parse "blank" tokens in hypothesis
#         if len(hypothesis.y_sequence) > 0 and hypothesis.y_sequence[-1] == self.blank_idx:
#             blank_state = True
#         else:
#             blank_state = False

#         # Convert last token of hypothesis to torch.Tensor
#         target = torch.full([1, 1], fill_value=hypothesis.y_sequence[-1], device=device, dtype=torch.long)
#         lm_token = target[:, -1]  # [1]

#         # Convert current hypothesis into a tuple to preserve in cache
#         sequence = tuple(hypothesis.y_sequence)

#         if sequence in cache:
#             y, new_state = cache[sequence]
#         else:
#             # Obtain score for target token and new states
#             if blank_state:
#                 y, new_state = self.predict(None, state=None, add_sos=False, batch_size=1)  # [1, 1, H]

#             else:
#                 y, new_state = self.predict(
#                     target, state=hypothesis.dec_state, add_sos=False, batch_size=1
#                 )  # [1, 1, H]

#             y = y[:, -1:, :]  # Extract just last state : [1, 1, H]
#             cache[sequence] = (y, new_state)

#         return y, new_state, lm_token

#     def batch_score_hypothesis(
#         self, hypotheses: List[Hypothesis], cache: Dict[Tuple[int], Any], batch_states: List[torch.Tensor]
#     ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
#         """
#         Used for batched beam search algorithms. Similar to score_hypothesis method.

#         Args:
#             hypothesis: List of Hypotheses. Refer to Hypothesis.
#             cache: Dict which contains a cache to avoid duplicate computations.
#             batch_states: List of torch.Tensor which represent the states of the RNN for this batch.
#                 Each state is of shape [L, B, H]

#         Returns:
#             Returns a tuple (b_y, b_states, lm_tokens) such that:
#             b_y is a torch.Tensor of shape [B, 1, H] representing the scores of the last tokens in the Hypotheses.
#             b_state is a list of list of RNN states, each of shape [L, B, H].
#             Represented as B x List[states].
#             lm_token is a list of the final integer tokens of the hypotheses in the batch.
#         """
#         final_batch = len(hypotheses)

#         if final_batch == 0:
#             raise ValueError("No hypotheses was provided for the batch!")

#         _p = next(self.parameters())
#         device = _p.device
#         dtype = _p.dtype

#         tokens = []
#         process = []
#         done = [None for _ in range(final_batch)]

#         # For each hypothesis, cache the last token of the sequence and the current states
#         for i, hyp in enumerate(hypotheses):
#             sequence = tuple(hyp.y_sequence)

#             if sequence in cache:
#                 done[i] = cache[sequence]
#             else:
#                 tokens.append(hyp.y_sequence[-1])
#                 process.append((sequence, hyp.dec_state))

#         if process:
#             batch = len(process)

#             # convert list of tokens to torch.Tensor, then reshape.
#             tokens = torch.tensor(tokens, device=device, dtype=torch.long).view(batch, -1)
#             dec_states = self.initialize_state(tokens.to(dtype=dtype))  # [L, B, H]
#             dec_states = self.batch_initialize_states(dec_states, [d_state for seq, d_state in process])

#             y, dec_states = self.predict(
#                 tokens, state=dec_states, add_sos=False, batch_size=batch
#             )  # [B, 1, H], List([L, 1, H])

#             dec_states = tuple(state.to(dtype=dtype) for state in dec_states)

#         # Update done states and cache shared by entire batch.
#         j = 0
#         for i in range(final_batch):
#             if done[i] is None:
#                 # Select sample's state from the batch state list
#                 new_state = self.batch_select_state(dec_states, j)

#                 # Cache [1, H] scores of the current y_j, and its corresponding state
#                 done[i] = (y[j], new_state)
#                 cache[process[j][0]] = (y[j], new_state)

#                 j += 1

#         # Set the incoming batch states with the new states obtained from `done`.
#         batch_states = self.batch_initialize_states(batch_states, [d_state for y_j, d_state in done])

#         # Create batch of all output scores
#         # List[1, 1, H] -> [B, 1, H]
#         batch_y = torch.stack([y_j for y_j, d_state in done])

#         # Extract the last tokens from all hypotheses and convert to a tensor
#         lm_tokens = torch.tensor([h.y_sequence[-1] for h in hypotheses], device=device, dtype=torch.long).view(
#             final_batch
#         )

#         return batch_y, batch_states, lm_tokens

#     def batch_initialize_states(self, batch_states: List[torch.Tensor], decoder_states: List[List[torch.Tensor]]):
#         """
#         Create batch of decoder states.

#        Args:
#            batch_states (list): batch of decoder states
#               ([L x (B, H)], [L x (B, H)])

#            decoder_states (list of list): list of decoder states
#                [B x ([L x (1, H)], [L x (1, H)])]

#        Returns:
#            batch_states (tuple): batch of decoder states
#                ([L x (B, H)], [L x (B, H)])
#        """
#         # LSTM has 2 states
#         new_states = [[] for _ in range(len(decoder_states[0]))]
#         for layer in range(self.pred_rnn_layers):
#             for state_id in range(len(decoder_states[0])):
#                 # batch_states[state_id][layer] = torch.stack([s[state_id][layer] for s in decoder_states])
#                 new_state_for_layer = torch.stack([s[state_id][layer] for s in decoder_states])
#                 new_states[state_id].append(new_state_for_layer)

#         for state_id in range(len(decoder_states[0])):
#             new_states[state_id] = torch.stack([state for state in new_states[state_id]])

#         return new_states

#     def batch_select_state(self, batch_states: List[torch.Tensor], idx: int) -> List[List[torch.Tensor]]:
#         """Get decoder state from batch of states, for given id.

#         Args:
#             batch_states (list): batch of decoder states
#                 ([L x (B, H)], [L x (B, H)])

#             idx (int): index to extract state from batch of states

#         Returns:
#             (tuple): decoder states for given id
#                 ([L x (1, H)], [L x (1, H)])
#         """
#         if batch_states is not None:
#             state_list = []
#             for state_id in range(len(batch_states)):
#                 states = [batch_states[state_id][layer][idx] for layer in range(self.pred_rnn_layers)]
#                 state_list.append(states)

#             return state_list
#         else:
#             return None

#     def batch_concat_states(self, batch_states: List[List[torch.Tensor]]) -> List[torch.Tensor]:
#         """Concatenate a batch of decoder state to a packed state.

#         Args:
#             batch_states (list): batch of decoder states
#                 B x ([L x (H)], [L x (H)])

#         Returns:
#             (tuple): decoder states
#                 (L x B x H, L x B x H)
#         """
#         state_list = []

#         for state_id in range(len(batch_states[0])):
#             batch_list = []
#             for sample_id in range(len(batch_states)):
#                 tensor = torch.stack(batch_states[sample_id][state_id])  # [L, H]
#                 tensor = tensor.unsqueeze(0)  # [1, L, H]
#                 batch_list.append(tensor)

#             state_tensor = torch.cat(batch_list, 0)  # [B, L, H]
#             state_tensor = state_tensor.transpose(1, 0)  # [L, B, H]
#             state_list.append(state_tensor)

#         return state_list

#     @classmethod
#     def batch_replace_states_mask(
#         cls,
#         src_states: Tuple[torch.Tensor, torch.Tensor],
#         dst_states: Tuple[torch.Tensor, torch.Tensor],
#         mask: torch.Tensor,
#     ):
#         """Replace states in dst_states with states from src_states using the mask"""
#         # same as `dst_states[i][mask] = src_states[i][mask]`, but non-blocking
#         # we need to cast, since LSTM is calculated in fp16 even if autocast to bfloat16 is enabled
#         dtype = dst_states[0].dtype
#         torch.where(mask.unsqueeze(0).unsqueeze(-1), src_states[0].to(dtype), dst_states[0], out=dst_states[0])
#         torch.where(mask.unsqueeze(0).unsqueeze(-1), src_states[1].to(dtype), dst_states[1], out=dst_states[1])

#     def batch_split_states(
#         self, batch_states: Tuple[torch.Tensor, torch.Tensor]
#     ) -> list[Tuple[torch.Tensor, torch.Tensor]]:
#         """
#         Split states into a list of states.
#         Useful for splitting the final state for converting results of the decoding algorithm to Hypothesis class.
#         """
#         return list(zip(batch_states[0].split(1, dim=1), batch_states[1].split(1, dim=1)))

#     def batch_copy_states(
#         self,
#         old_states: List[torch.Tensor],
#         new_states: List[torch.Tensor],
#         ids: List[int],
#         value: Optional[float] = None,
#     ) -> List[torch.Tensor]:
#         """Copy states from new state to old state at certain indices.

#         Args:
#             old_states(list): packed decoder states
#                 (L x B x H, L x B x H)

#             new_states: packed decoder states
#                 (L x B x H, L x B x H)

#             ids (list): List of indices to copy states at.

#             value (optional float): If a value should be copied instead of a state slice, a float should be provided

#         Returns:
#             batch of decoder states with partial copy at ids (or a specific value).
#                 (L x B x H, L x B x H)
#         """
#         for state_id in range(len(old_states)):
#             if value is None:
#                 old_states[state_id][:, ids, :] = new_states[state_id][:, ids, :]
#             else:
#                 old_states[state_id][:, ids, :] *= 0.0
#                 old_states[state_id][:, ids, :] += value

#         return old_states

#     def mask_select_states(
#         self, states: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Return states by mask selection
#         Args:
#             states: states for the batch
#             mask: boolean mask for selecting states; batch dimension should be the same as for states

#         Returns:
#             states filtered by mask
#         """
#         # LSTM in PyTorch returns a tuple of 2 tensors as a state
#         return states[0][:, mask], states[1][:, mask]

# class RNNTJoint(nn.Module):

#     def __init__(
#         self,
#         jointnet: Dict[str, Any],
#         num_classes: int,
#         num_extra_outputs: int = 0,
#         vocabulary: Optional[List] = None,
#         log_softmax: Optional[bool] = None,
#         preserve_memory: bool = False,
#         fuse_loss_wer: bool = False,
#         fused_batch_size: Optional[int] = None,
#         experimental_fuse_loss_wer: Any = None,
#         language_masks=None, #CTEMO
#         multilingual: bool = False, #CTEMO
#         language_keys: Optional[List] = None, #CTEMO
#         token_id_offsets=None, #CTEMO
#         offset_token_ids_by_token_id=None, #CTEMO
#     ):
#         super().__init__()

#         self.vocabulary = vocabulary

#         self._vocab_size = num_classes
#         self._num_extra_outputs = num_extra_outputs
#         self._num_classes = num_classes + 1 + num_extra_outputs  # 1 is for blank
#         self.language_masks = language_masks #CTEMO
#         self.token_id_offsets = token_id_offsets #CTEMO
#         self.offset_token_ids_by_token_id = offset_token_ids_by_token_id #CTEMO
#         self.multilingual = multilingual #CTEMO
#         self.language_keys = language_keys #CTEMO

#         if experimental_fuse_loss_wer is not None:
#             # Override fuse_loss_wer from deprecated argument
#             fuse_loss_wer = experimental_fuse_loss_wer

#         self._fuse_loss_wer = fuse_loss_wer
#         self._fused_batch_size = fused_batch_size

#         if fuse_loss_wer and (fused_batch_size is None):
#             raise ValueError("If `fuse_loss_wer` is set, then `fused_batch_size` cannot be None!")

#         self._loss = None
#         self._wer = None

#         # Log softmax should be applied explicitly only for CPU
#         self.log_softmax = log_softmax
#         self.preserve_memory = preserve_memory

#         if preserve_memory:
#             logging.warning(
#                 "`preserve_memory` was set for the Joint Model. Please be aware this will severely impact "
#                 "the forward-backward step time. It also might not solve OOM issues if the GPU simply "
#                 "does not have enough memory to compute the joint."
#             )

#         # Required arguments
#         self.encoder_hidden = jointnet['encoder_hidden']
#         self.pred_hidden = jointnet['pred_hidden']
#         self.joint_hidden = jointnet['joint_hidden']
#         self.activation = jointnet['activation']

#         # Optional arguments
#         dropout = jointnet.get('dropout', 0.0)

#         self.pred, self.enc, self.joint_net = self._joint_net_modules(
#             num_classes=self._num_classes,  # add 1 for blank symbol
#             pred_n_hidden=self.pred_hidden,
#             enc_n_hidden=self.encoder_hidden,
#             joint_n_hidden=self.joint_hidden,
#             activation=self.activation,
#             dropout=dropout,
#         )

#         # Flag needed for RNNT export support
#         self._rnnt_export = False

#         # to change, requires running ``model.temperature = T`` explicitly
#         self.temperature = 1.0

#     def forward(
#         self,
#         encoder_outputs: torch.Tensor,
#         decoder_outputs: Optional[torch.Tensor],
#         encoder_lengths: Optional[torch.Tensor] = None,
#         transcripts: Optional[torch.Tensor] = None,
#         transcript_lengths: Optional[torch.Tensor] = None,
#         compute_wer: bool = False,
#         language_ids=None, #CTEMO
#     ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
#         # encoder = (B, D, T)
#         # decoder = (B, D, U) if passed, else None
#         encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)

#         if decoder_outputs is not None:
#             decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

#         if not self._fuse_loss_wer:
#             if decoder_outputs is None:
#                 raise ValueError(
#                     "decoder_outputs passed is None, and `fuse_loss_wer` is not set. "
#                     "decoder_outputs can only be None for fused step!"
#                 )

#             out = self.joint(encoder_outputs, decoder_outputs, language_ids=language_ids)  # [B, T, U, V + 1] #CTEMO
#             return out

#         else:
#             # At least the loss module must be supplied during fused joint
#             if self._loss is None or self._wer is None:
#                 raise ValueError("`fuse_loss_wer` flag is set, but `loss` and `wer` modules were not provided! ")

#             # If fused joint step is required, fused batch size is required as well
#             if self._fused_batch_size is None:
#                 raise ValueError("If `fuse_loss_wer` is set, then `fused_batch_size` cannot be None!")

#             # When using fused joint step, both encoder and transcript lengths must be provided
#             if (encoder_lengths is None) or (transcript_lengths is None):
#                 raise ValueError(
#                     "`fuse_loss_wer` is set, therefore encoder and target lengths " "must be provided as well!"
#                 )

#             losses = []
#             wers, wer_nums, wer_denoms = [], [], []
#             target_lengths = []
#             batch_size = int(encoder_outputs.size(0))  # actual batch size

#             # Iterate over batch using fused_batch_size steps
#             for batch_idx in range(0, batch_size, self._fused_batch_size):
#                 begin = batch_idx
#                 end = min(begin + self._fused_batch_size, batch_size)

#                 # Extract the sub batch inputs
#                 # sub_enc = encoder_outputs[begin:end, ...]
#                 # sub_transcripts = transcripts[begin:end, ...]
#                 sub_enc = encoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))
#                 sub_transcripts = transcripts.narrow(dim=0, start=begin, length=int(end - begin))

#                 sub_enc_lens = encoder_lengths[begin:end]
#                 sub_transcript_lens = transcript_lengths[begin:end]

#                 # Sub transcripts does not need the full padding of the entire batch
#                 # Therefore reduce the decoder time steps to match
#                 max_sub_enc_length = sub_enc_lens.max()
#                 max_sub_transcript_length = sub_transcript_lens.max()

#                 if decoder_outputs is not None:
#                     # Reduce encoder length to preserve computation
#                     # Encoder: [sub-batch, T, D] -> [sub-batch, T', D]; T' < T
#                     if sub_enc.shape[1] != max_sub_enc_length:
#                         sub_enc = sub_enc.narrow(dim=1, start=0, length=int(max_sub_enc_length))

#                     # sub_dec = decoder_outputs[begin:end, ...]  # [sub-batch, U, D]
#                     sub_dec = decoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))  # [sub-batch, U, D]

#                     # Reduce decoder length to preserve computation
#                     # Decoder: [sub-batch, U, D] -> [sub-batch, U', D]; U' < U
#                     if sub_dec.shape[1] != max_sub_transcript_length + 1:
#                         sub_dec = sub_dec.narrow(dim=1, start=0, length=int(max_sub_transcript_length + 1))

#                     # Perform joint => [sub-batch, T', U', V + 1]
#                     if language_ids is not None: #CTEMO
#                         sub_joint = self.joint(sub_enc, sub_dec, language_ids=language_ids[begin:end]) #CTEMO
#                     else:
#                         sub_joint = self.joint(sub_enc, sub_dec) #CTEMO

#                     del sub_dec

#                     # Reduce transcript length to correct alignment
#                     # Transcript: [sub-batch, L] -> [sub-batch, L']; L' <= L
#                     if sub_transcripts.shape[1] != max_sub_transcript_length:
#                         sub_transcripts = sub_transcripts.narrow(dim=1, start=0, length=int(max_sub_transcript_length))

#                     # Compute sub batch loss
#                     # preserve loss reduction type
#                     loss_reduction = self.loss.reduction

#                     # override loss reduction to sum
#                     self.loss.reduction = None

#                     # compute and preserve loss
#                     loss_batch = self.loss(
#                         log_probs=sub_joint,
#                         targets=sub_transcripts,
#                         input_lengths=sub_enc_lens,
#                         target_lengths=sub_transcript_lens,
#                     )
#                     losses.append(loss_batch)
#                     target_lengths.append(sub_transcript_lens)

#                     # reset loss reduction type
#                     self.loss.reduction = loss_reduction

#                 else:
#                     losses = None

#                 # Update WER for sub batch
#                 if compute_wer:
#                     sub_enc = sub_enc.transpose(1, 2)  # [B, T, D] -> [B, D, T]
#                     sub_enc = sub_enc.detach()
#                     sub_transcripts = sub_transcripts.detach()

#                     # Update WER on each process without syncing
#                     if language_ids is not None: #CTEMO
#                         self.wer.update(
#                             predictions=sub_enc,
#                             predictions_lengths=sub_enc_lens,
#                             targets=sub_transcripts,
#                             targets_lengths=sub_transcript_lens,
#                             lang_ids=language_ids[begin:end]
#                         )
#                     else:
#                         self.wer.update(
#                             predictions=sub_enc,
#                             predictions_lengths=sub_enc_lens,
#                             targets=sub_transcripts,
#                             targets_lengths=sub_transcript_lens,
#                         )
#                     # Sync and all_reduce on all processes, compute global WER
#                     wer, wer_num, wer_denom = self.wer.compute()
#                     self.wer.reset()

#                     wers.append(wer)
#                     wer_nums.append(wer_num)
#                     wer_denoms.append(wer_denom)

#                 del sub_enc, sub_transcripts, sub_enc_lens, sub_transcript_lens

#             # Reduce over sub batches
#             if losses is not None:
#                 losses = self.loss.reduce(losses, target_lengths)

#             # Collect sub batch wer results
#             if compute_wer:
#                 wer = sum(wers) / len(wers)
#                 wer_num = sum(wer_nums)
#                 wer_denom = sum(wer_denoms)
#             else:
#                 wer = None
#                 wer_num = None
#                 wer_denom = None

#             return losses, wer, wer_num, wer_denom

#     def joint(self, f: torch.Tensor, g: torch.Tensor, language_ids=None) -> torch.Tensor:
#         return self.joint_after_projection(self.project_encoder(f), self.project_prednet(g), language_ids)

#     def project_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
#         """
#         Project the encoder output to the joint hidden dimension.

#         Args:
#             encoder_output: A torch.Tensor of shape [B, T, D]

#         Returns:
#             A torch.Tensor of shape [B, T, H]
#         """
#         return self.enc(encoder_output)

#     def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
#         """
#         Project the Prediction Network (Decoder) output to the joint hidden dimension.

#         Args:
#             prednet_output: A torch.Tensor of shape [B, U, D]

#         Returns:
#             A torch.Tensor of shape [B, U, H]
#         """
#         return self.pred(prednet_output)

#     def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor, language_ids=None) -> torch.Tensor: #CTEMO
#         """
#         Compute the joint step of the network after projection.

#         Here,
#         B = Batch size
#         T = Acoustic model timesteps
#         U = Target sequence length
#         H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
#         H = Hidden dimension of the Joint hidden step.
#         V = Vocabulary size of the Decoder (excluding the RNNT blank token).

#         NOTE:
#             The implementation of this model is slightly modified from the original paper.
#             The original paper proposes the following steps :
#             (enc, dec) -> Expand + Concat + Sum [B, T, U, H1+H2] -> Forward through joint hidden [B, T, U, H] -- *1
#             *1 -> Forward through joint final [B, T, U, V + 1].

#             We instead split the joint hidden into joint_hidden_enc and joint_hidden_dec and act as follows:
#             enc -> Forward through joint_hidden_enc -> Expand [B, T, 1, H] -- *1
#             dec -> Forward through joint_hidden_dec -> Expand [B, 1, U, H] -- *2
#             (*1, *2) -> Sum [B, T, U, H] -> Forward through joint final [B, T, U, V + 1].

#         Args:
#             f: Output of the Encoder model. A torch.Tensor of shape [B, T, H1]
#             g: Output of the Decoder model. A torch.Tensor of shape [B, U, H2]

#         Returns:
#             Logits / log softmaxed tensor of shape (B, T, U, V + 1).
#         """
#         f = f.unsqueeze(dim=2)  # (B, T, 1, H)
#         g = g.unsqueeze(dim=1)  # (B, 1, U, H)
#         inp = f + g  # [B, T, U, H]

#         del f, g

#         # Forward adapter modules on joint hidden
#         # if self.is_adapter_available():
#         #     inp = self.forward_enabled_adapters(inp)

#         if language_ids is not None: #CTEMO
#             # Do partial forward of joint net (skipping the final linear)
#             for module in self.joint_net[:-1]:
#                 inp = module(inp)  # [B, T, U, H]
            
#             # check if all the items in the batch have the same langauge, pass them through
#             if len(set(language_ids)) == 1:
#                 res = self.joint_net[-1][language_ids[0]](inp)
#             else:
#                 res_single = []
#                 for single_inp, lang in zip(inp, language_ids):
#                     res_single.append(self.joint_net[-1][lang](single_inp))
#                 res = torch.stack(res_single)
#         else:
#             res = self.joint_net(inp)  # [B, T, U, V + 1]

#         del inp

#         if self.preserve_memory:
#             torch.cuda.empty_cache()

#         # If log_softmax is automatic
#         if self.log_softmax is None:
#             if not res.is_cuda:  # Use log softmax only if on CPU
#                 if self.temperature != 1.0:
#                     res = (res / self.temperature).log_softmax(dim=-1)
#                 else:
#                     res = res.log_softmax(dim=-1)
#         else:
#             if self.log_softmax:
#                 if self.temperature != 1.0:
#                     res = (res / self.temperature).log_softmax(dim=-1)
#                 else:
#                     res = res.log_softmax(dim=-1)

#         return res

#     def _joint_net_modules(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout):
#         """
#         Prepare the trainable modules of the Joint Network

#         Args:
#             num_classes: Number of output classes (vocab size) excluding the RNNT blank token.
#             pred_n_hidden: Hidden size of the prediction network.
#             enc_n_hidden: Hidden size of the encoder network.
#             joint_n_hidden: Hidden size of the joint network.
#             activation: Activation of the joint. Can be one of [relu, tanh, sigmoid]
#             dropout: Dropout value to apply to joint.
#         """
#         pred = nn.Linear(pred_n_hidden, joint_n_hidden)
#         enc = nn.Linear(enc_n_hidden, joint_n_hidden)

#         if activation not in ['relu', 'sigmoid', 'tanh']:
#             raise ValueError("Unsupported activation for joint step - please pass one of " "[relu, sigmoid, tanh]")

#         activation = activation.lower()

#         if activation == 'relu':
#             activation = nn.ReLU(inplace=True)
#         elif activation == 'sigmoid':
#             activation = nn.Sigmoid()
#         elif activation == 'tanh':
#             activation = nn.Tanh()

#         if self.multilingual: #CTEMO
#             final_layer = nn.ModuleDict()
#             logging.info(f"Vocab size for each language: {self._vocab_size // len(self.language_keys)}")
#             for lang in self.language_keys:
#                 final_layer[lang] = nn.Linear(joint_n_hidden, (self._vocab_size // len(self.language_keys)+1))
#             layers = (
#                 [activation]
#                 + ([nn.Dropout(p=dropout)] if dropout else [])
#                 + [final_layer]
#             )
#         else:
#             layers = (
#                 [activation]
#                 + ([nn.Dropout(p=dropout)] if dropout else [])
#                 + [nn.Linear(joint_n_hidden, num_classes)]
#             )
#         return pred, enc, nn.Sequential(*layers)

#     @property
#     def num_classes_with_blank(self):
#         return self._num_classes

#     @property
#     def num_extra_outputs(self):
#         return self._num_extra_outputs

#     @property
#     def loss(self):
#         return self._loss

#     def set_loss(self, loss):
#         if not self._fuse_loss_wer:
#             raise ValueError("Attempting to set loss module even though `fuse_loss_wer` is not set!")

#         self._loss = loss

#     @property
#     def wer(self):
#         return self._wer

#     def set_wer(self, wer):
#         if not self._fuse_loss_wer:
#             raise ValueError("Attempting to set WER module even though `fuse_loss_wer` is not set!")

#         self._wer = wer

#     @property
#     def fuse_loss_wer(self):
#         return self._fuse_loss_wer

#     def set_fuse_loss_wer(self, fuse_loss_wer, loss=None, metric=None):
#         self._fuse_loss_wer = fuse_loss_wer

#         self._loss = loss
#         self._wer = metric

#     @property
#     def fused_batch_size(self):
#         return self._fused_batch_size

#     def set_fused_batch_size(self, fused_batch_size):
#         self._fused_batch_size = fused_batch_size

# class GreedyBatchedRNNTInfer:
#     def __init__(
#         self,
#         decoder: RNNTDecoder,
#         joint: RNNTJoint,
#         blank_index: int,
#         max_symbols_per_step: Optional[int] = None,
#     ):
#         super().__init__()

#         self._blank_index = blank_index
#         self.max_symbols = max_symbols_per_step
#         self._SOS = blank_index
#         self.decoder = decoder
#         self.joint = joint

#     def _pred_step(
#         self,
#         label: Union[torch.Tensor, int],
#         hidden: Optional[torch.Tensor],
#         add_sos: bool = False,
#         batch_size: Optional[int] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         if isinstance(label, torch.Tensor):
#             if label.dtype != torch.long:
#                 label = label.long()
#         else:
#             if label == self._SOS:
#                 return self.decoder.predict(None, hidden, add_sos=add_sos, batch_size=batch_size)
#             label = label_collate([[label]])
#         return self.decoder.predict(label, hidden, add_sos=add_sos, batch_size=batch_size)

#     def _joint_step(self, enc:torch.Tensor, pred:torch.Tensor, log_normalize: Optional[bool] = None, language_ids=None): # CTEMO
#         self.joint._fuse_loss_wer = False
#         logits = self.joint(encoder_outputs=enc.transpose(1, 2), decoder_outputs=pred.transpose(1, 2), language_ids=language_ids)
#         self.joint._fuse_loss_wer = True
#         if not logits.is_cuda:  # Use log softmax only if on CPU
#             logits = logits.log_softmax(dim=len(logits.shape) - 1)
#         return logits

#     def __call__(
#         self,
#         x: torch.Tensor,
#         out_len: torch.Tensor,
#         language_ids=None,
#     ):
#         batchsize = x.shape[0]
#         hypotheses = [Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)]
#         hidden = None
#         last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long)
#         blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool)
#         blank_mask_prev = None
#         max_out_len = out_len.max()
#         for time_idx in range(max_out_len):
#             f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]
#             not_blank = True
#             symbols_added = 0
#             blank_mask.mul_(False)
#             blank_mask = time_idx >= out_len
#             blank_mask_prev = blank_mask.clone()
#             while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
#                 if time_idx == 0 and symbols_added == 0 and hidden is None:
#                     g, hidden_prime = self._pred_step(self._SOS, hidden, batch_size=batchsize)
#                 else:
#                     g, hidden_prime = self._pred_step(last_label, hidden, batch_size=batchsize)
#                 logp = self._joint_step(f, g, language_ids=language_ids)[:, 0, 0, :]
#                 if logp.dtype != torch.float32:
#                     logp = logp.float()
#                 v, k = logp.max(1)
#                 del g
#                 k_is_blank = k == self._blank_index
#                 blank_mask.bitwise_or_(k_is_blank)
#                 del k_is_blank
#                 del logp
#                 blank_mask_prev.bitwise_or_(blank_mask)
#                 if blank_mask.all():
#                     not_blank = False
#                 else:
#                     blank_indices = (blank_mask == 1).nonzero(as_tuple=False)
#                     if hidden is not None:
#                         hidden_prime = self.decoder.batch_copy_states(hidden_prime, hidden, blank_indices)
#                     elif len(blank_indices) > 0 and hidden is None:
#                         hidden_prime = self.decoder.batch_copy_states(hidden_prime, None, blank_indices, value=0.0)
#                     k[blank_indices] = last_label[blank_indices, 0]
#                     last_label = k.clone().view(-1, 1)
#                     hidden = hidden_prime
#                     for kidx, ki in enumerate(k):
#                         if blank_mask[kidx] == 0:
#                             hypotheses[kidx].y_sequence.append(ki)
#                             hypotheses[kidx].timestep.append(time_idx)
#                             hypotheses[kidx].score += float(v[kidx])
#                     symbols_added += 1
#         for batch_idx in range(batchsize):
#             hypotheses[batch_idx].dec_state = self.decoder.batch_select_state(hidden, batch_idx)
#         return hypotheses




class MelPreprocessor(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=512,
        n_mels=80,
        preemph=0.97,
        log=True,
        log_zero_guard_value=2 ** -24,
        normalize="per_feature",
        pad_to=16,
        pad_value=0.0,
        mag_power=2.0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.preemph = preemph
        self.log = log
        self.log_zero_guard_value = log_zero_guard_value
        self.normalize = normalize
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.mag_power = mag_power

        # Window (NeMo uses periodic=False)
        window = torch.hann_window(self.win_length, periodic=False)
        self.register_buffer("window", window)

        # Precomputed NeMo mel filterbank (librosa-based)
        mel_fb = torch.tensor(librosa.filters.mel(sr=sample_rate,n_fft=n_fft,n_mels=n_mels), dtype=torch.float32)
        self.register_buffer("mel_fb", mel_fb)

    def _preemphasis(self, x):
        if self.preemph == 0.0:
            return x
        return torch.cat(
            (x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1
        )

    def _get_feat_len(self, lengths):
        # NeMo formula (center=True)
        pad = self.n_fft
        return torch.floor_divide(lengths + pad - self.n_fft, self.hop_length) + 1

    def forward(self, waveform, lengths):
        """
        waveform: [B, T]
        lengths:  [B]
        returns:
            feats: [B, n_mels, T']
            feat_lens: [B]
        """

        # Preemphasis
        x = self._preemphasis(waveform)

        # STFT (NeMo-style)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=x.dtype),
            center=True,
            return_complex=True,
        )

        # Magnitude
        spec = torch.view_as_real(spec)
        mag = torch.sqrt(spec.pow(2).sum(-1) + CONSTANT)

        # Power
        if self.mag_power != 1.0:
            mag = mag.pow(self.mag_power)

        # Mel projection (NeMo order)
        feats = torch.matmul(self.mel_fb.to(mag.dtype), mag)

        # Log
        if self.log:
            feats = torch.log(feats + self.log_zero_guard_value)

        # Feature lengths
        feat_lens = self._get_feat_len(lengths).to(torch.long)

        # Normalization (NeMo-style)
        if self.normalize == "per_feature":
            B, C, T = feats.shape
            mask = (
                torch.arange(T, device=feats.device)
                .unsqueeze(0)
                .expand(B, T)
                >= feat_lens.unsqueeze(1)
            )

            feats = feats.masked_fill(mask.unsqueeze(1), 0.0)
            mean = feats.sum(-1) / feat_lens.unsqueeze(1)
            var = ((feats - mean.unsqueeze(-1)) ** 2).sum(-1) / (feat_lens.unsqueeze(1) - 1)
            std = torch.sqrt(var + CONSTANT)

            feats = (feats - mean.unsqueeze(-1)) / std.unsqueeze(-1)
            feats = feats.masked_fill(mask.unsqueeze(1), 0.0)

        # Pad to multiple of pad_to
        if self.pad_to > 0:
            pad_amt = feats.shape[-1] % self.pad_to
            if pad_amt != 0:
                feats = F.pad(
                    feats,
                    (0, self.pad_to - pad_amt),
                    value=self.pad_value,
                )

        return feats, feat_lens

class ConvSubsampling(nn.Module):

    def __init__(
        self,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(True),
    ):
        super().__init__()
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor
        in_channels = 1
        layers = []
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False
        self._left_padding = (self._kernel_size - 1) // 2
        self._right_padding = (self._kernel_size - 1) // 2
        self._max_cache_len = 0
        layers.append(nn.Conv2d(in_channels,conv_channels,self._kernel_size,self._stride,self._left_padding))
        in_channels = conv_channels
        layers.append(activation)
        for _ in range(self._sampling_num - 1):
            layers.extend([nn.Conv2d(in_channels,in_channels,self._kernel_size,self._stride,self._left_padding,groups=in_channels,),nn.Conv2d(in_channels,conv_channels,1,1),activation])
            in_channels = conv_channels
        self.out = nn.Linear(conv_channels * int(calc_length(
            lengths=torch.tensor(feat_in, dtype=torch.float),
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )), feat_out)
        self.conv2d_subsampling = True
        self.conv = nn.Sequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def forward(self, x:torch.Tensor, lengths:torch.Tensor):

        # Unsqueeze Channel Axis
        if self.conv2d_subsampling:
            x = x.unsqueeze(1)
        # Transpose to Channel First mode
        else:
            x = x.transpose(1, 2)

        # split inputs if chunking_factor is set
        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:
                # if subsampling_conv_chunking_factor is 1, we split only if needed
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                x_ceil = 2 ** 31 / self._conv_channels * self._stride * self._stride
                if torch.numel(x) > x_ceil:
                    need_to_split = True
                else:
                    need_to_split = False
            else:
                # if subsampling_conv_chunking_factor > 1 we always split
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:  # if unable to split by batch, try by channel
                    x = self.conv_split_by_channel(x)
            else:
                x = self.conv(x)
        else:
            x = self.conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)

        return x, calc_length(lengths,self._left_padding + self._right_padding,self._kernel_size,self._stride,self._ceil_mode,self._sampling_num,)

    def reset_parameters(self):
        # initialize weights
        with torch.no_grad():
            # init conv
            scale = 1.0 / self._kernel_size
            dw_max = (self._kernel_size ** 2) ** -0.5
            pw_max = self._conv_channels ** -0.5

            nn.init.uniform_(self.conv[0].weight, -scale, scale)
            nn.init.uniform_(self.conv[0].bias, -scale, scale)

            for idx in range(2, len(self.conv), 3):
                nn.init.uniform_(self.conv[idx].weight, -dw_max, dw_max)
                nn.init.uniform_(self.conv[idx].bias, -dw_max, dw_max)
                nn.init.uniform_(self.conv[idx + 1].weight, -pw_max, pw_max)
                nn.init.uniform_(self.conv[idx + 1].bias, -pw_max, pw_max)

            # init fc (80 * 64 = 5120 from https://github.com/kssteven418/Squeezeformer/blob/13c97d6cf92f2844d2cb3142b4c5bfa9ad1a8951/src/models/conformer_encoder.py#L487
            fc_scale = (self._feat_out * self._feat_in / self._sampling_num) ** -0.5
            nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
            nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)

    def conv_split_by_batch(self, x:torch.Tensor):
        """ Tries to split input by batch, run conv and concat results """
        b, _, _, _ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
            logging.debug(f'using manually set chunking factor: {cf}')
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2 ** 31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
            cf = 2 ** p
            logging.debug(f'using auto set chunking factor: {cf}')

        new_batch_size = b // cf
        if new_batch_size == 0:  # input is too big
            return x, False

        logging.debug(f'conv subsampling: using split batch size {new_batch_size}')
        return torch.cat([self.conv(chunk) for chunk in torch.split(x, new_batch_size, 0)]), True

    def conv_split_by_channel(self, x):
        """ For dw convs, tries to split input by time, run conv and concat results """
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
                logging.debug(f'using manually set chunking factor: {cf}')
            else:
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                p = math.ceil(math.log(torch.numel(x) / 2 ** 31, 2))
                cf = 2 ** p
                logging.debug(f'using auto set chunking factor: {cf}')

            new_c = int(c // cf)
            if new_c == 0:
                logging.warning(f'chunking factor {cf} is too high; splitting down to one channel.')
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                logging.warning(f'chunking factor {cf} is too high; splitting down to one timestep.')
                new_t = 1

            logging.debug(f'conv dw subsampling: using split C size {new_c} and split T size {new_t}')
            x = self.channel_chunked_conv(self.conv[i * 3 + 2], new_c, x)  # conv2D, depthwise

            # splitting pointwise convs by time
            x = torch.cat([self.conv[i * 3 + 3](chunk) for chunk in torch.split(x, new_t, 2)], 2)  # conv2D, pointwise
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """ Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(
                    chunk, pad=(self._kernel_size - 1, self._stride - 1, self._kernel_size - 1, self._stride - 1)
                )
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind : ind + step, :, :, :],
                    bias=conv.bias[ind : ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)

    def change_subsampling_conv_chunking_factor(self, subsampling_conv_chunking_factor: int):
        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("subsampling_conv_chunking_factor should be -1, 1, or a power of 2")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

def make_chunks(file_path,aggressiveness=2,min_chunk_sec=10,max_chunk_sec=15,frame_ms=30,format=None):
    wav, sr = torchaudio.load(file_path, normalize=True,format=format)
    wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = F_torchaudio.resample(wav, sr, 16000)
        sr = 16000

    wav_int16 = (wav * 32768).clamp(-32768, 32767).short().squeeze(0)
    total_samples = len(wav_int16)

    frame_len = int(sr * frame_ms / 1000)
    total_frames = len(wav_int16) // frame_len
    wav_int16 = wav_int16[: total_frames * frame_len]
    frames = wav_int16.view(total_frames, frame_len)

    vad = webrtcvad.Vad(aggressiveness)
    is_speech = torch.zeros(total_frames, dtype=torch.bool)

    for i, f in enumerate(frames):
        try:
            is_speech[i] = vad.is_speech(f.numpy().tobytes(), sr)
        except:
            is_speech[i] = False

    segs, start_idx = [], None
    for i, s in enumerate(is_speech):
        if s and start_idx is None:
            start_idx = i
        elif not s and start_idx is not None:
            segs.append((start_idx, i))
            start_idx = None
    if start_idx is not None:
        segs.append((start_idx, len(is_speech)))
    chunks = []
    times_list = []
    start_sample = 0
    min_len = int(min_chunk_sec * sr)
    max_len = int(max_chunk_sec * sr)
    while start_sample < total_samples:
        end_sample = min(start_sample + max_len, total_samples)
        chunk_end_frame = end_sample // frame_len
        while chunk_end_frame < len(is_speech) and is_speech[chunk_end_frame]:
            chunk_end_frame += 1
            end_sample = min(chunk_end_frame * frame_len, total_samples)
            if end_sample - start_sample > max_len * 1.5:
                break
        if end_sample - start_sample < min_len and end_sample < total_samples:
            end_sample = min(start_sample + min_len, total_samples)
        chunk = wav[:, start_sample:end_sample]
        chunks.append(chunk.squeeze())
        times_list.append([
            round(start_sample / sr, 2),
            round(end_sample / sr, 2)
        ])
        start_sample = end_sample
    times = torch.tensor(times_list, dtype=torch.float32)
    return chunks, times

def make_srt(h: list[Hypothesis], ts: torch.Tensor, tokenizer, denormalizer: int=0.08): # 0.08 is making because of subsampling_factor * window stride = 8 * 0.01
    timestamp = []
    for hyp, (s, e) in zip(h, ts):
        starts = s + np.array(hyp.timestep) * denormalizer
        for y, st, en in zip(hyp.y_sequence,starts,list(starts[1:]) + [e]):
            timestamp.append({"text": tokenizer[int(y)],"start": float(st),"end": float(en)})
        timestamp.append({"text": "<line>","start": float(e),"end": float(e + 0.005)})
    return [
        srt.Subtitle(
            index,
            timedelta(seconds=item["start"]),
            timedelta(seconds=item["end"]),
            item["text"]
        )
        for index, item in enumerate(timestamp, 1)
    ]

def padding_audio(batch):
    audios, times = zip(*batch)

    lengths = torch.tensor([len(x) for x in audios])
    max_len = lengths.max()

    padded = torch.stack([
        F.pad(x, (0, max_len - len(x)))
        for x in audios
    ])

    times = torch.stack(times)   # each is a tensor [start, end]

    return padded, lengths, times

class ChunkedData(Dataset):
    def __init__(self, audio_path):
        self.data,self.ts = make_chunks(audio_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx],self.ts[idx]
