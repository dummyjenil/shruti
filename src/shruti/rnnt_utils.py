import logging
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

class LSTMDropout(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: Optional[float],
        forget_gate_bias: Optional[float],
        t_max: Optional[int] = None,
        weights_init_scale: float = 1.0,
        hidden_hidden_bias_scale: float = 0.0,
        proj_size: int = 0,
    ):
        """Returns an LSTM with forget gate bias init to `forget_gate_bias`.
        Args:
            input_size: See `nn.LSTM`.
            hidden_size: See `nn.LSTM`.
            num_layers: See `nn.LSTM`.
            dropout: See `nn.LSTM`.

            forget_gate_bias: float, set by default to 1.0, which constructs a forget gate
                initialized to 1.0.
                Reference:
                [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

            t_max: int value, set to None by default. If an int is specified, performs Chrono Initialization
                of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
                of training.
                Reference:
                [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)

            weights_init_scale: Float scale of the weights after initialization. Setting to lower than one
                sometimes helps reduce variance between runs.

            hidden_hidden_bias_scale: Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
                the default behaviour.

        Returns:
            A `nn.LSTM`.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, proj_size=proj_size
        )

        if t_max is not None:
            # apply chrono init
            for name, v in self.lstm.named_parameters():
                if 'bias' in name:
                    p = getattr(self.lstm, name)
                    n = p.nelement()
                    hidden_size = n // 4
                    p.data.fill_(0)
                    p.data[hidden_size : 2 * hidden_size] = torch.log(
                        nn.init.uniform_(p.data[0:hidden_size], 1, t_max - 1)
                    )
                    # forget gate biases = log(uniform(1, Tmax-1))
                    p.data[0:hidden_size] = -p.data[hidden_size : 2 * hidden_size]
                    # input gate biases = -(forget gate biases)

        elif forget_gate_bias is not None:
            for name, v in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size : 2 * hidden_size] *= float(hidden_hidden_bias_scale)

        self.dropout = nn.Dropout(dropout) if dropout else None

        for name, v in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                v.data *= float(weights_init_scale)

    def forward(
        self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, h = self.lstm(x, h)

        if self.dropout:
            x = self.dropout(x)

        return x, h

@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms.

    score: A float score obtained from an AbstractRNNTDecoder module's score_hypothesis method.

    y_sequence: Either a sequence of integer ids pointing to some vocabulary, or a packed torch.Tensor
        behaving in the same manner. dtype must be torch.Long in the latter case.

    dec_state: A list (or list of list) of LSTM-RNN decoder states. Can be None.

    text: (Optional) A decoded string after processing via CTC / RNN-T decoding (removing the CTC/RNNT
        `blank` tokens, and optionally merging word-pieces). Should be used as decoded string for
        Word Error Rate calculation.

    timestep: (Optional) A list of integer indices representing at which index in the decoding
        process did the token appear. Should be of same length as the number of non-blank tokens.

    alignments: (Optional) Represents the CTC / RNNT token alignments as integer tokens along an axis of
        time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of integer indices.
        For RNNT, represented as a dangling list of list of integer indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).
        The set of valid indices **includes** the CTC / RNNT blank token in order to represent alignments.

    frame_confidence: (Optional) Represents the CTC / RNNT per-frame confidence scores as token probabilities
        along an axis of time T (for CTC) or Time x Target (TxU).
        For CTC, represented as a single list of float indices.
        For RNNT, represented as a dangling list of list of float indices.
        Outer list represents Time dimension (T), inner list represents Target dimension (U).

    token_confidence: (Optional) Represents the CTC / RNNT per-token confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    word_confidence: (Optional) Represents the CTC / RNNT per-word confidence scores as token probabilities
        along an axis of Target U.
        Represented as a single list of float indices.

    length: Represents the length of the sequence (the original length without padding), otherwise
        defaults to 0.

    y: (Unused) A list of torch.Tensors representing the list of hypotheses.

    lm_state: (Unused) A dictionary state cache used by an external Language Model.

    lm_scores: (Unused) Score of the external Language Model.

    ngram_lm_state: (Optional) State of the external n-gram Language Model.

    tokens: (Optional) A list of decoded tokens (can be characters or word-pieces.

    last_token (Optional): A token or batch of tokens which was predicted in the last step.
    """

    score: float
    y_sequence: Union[List[int], torch.Tensor]
    text: Optional[str] = None
    dec_out: Optional[List[torch.Tensor]] = None
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    timestep: Union[List[int], torch.Tensor] = field(default_factory=list)
    alignments: Optional[Union[List[int], List[List[int]]]] = None
    frame_confidence: Optional[Union[List[float], List[List[float]]]] = None
    token_confidence: Optional[List[float]] = None
    word_confidence: Optional[List[float]] = None
    length: Union[int, torch.Tensor] = 0
    y: List[torch.tensor] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    lm_scores: Optional[torch.Tensor] = None
    ngram_lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    tokens: Optional[Union[List[int], torch.Tensor]] = None
    last_token: Optional[torch.Tensor] = None

    @property
    def non_blank_frame_confidence(self) -> List[float]:
        """Get per-frame confidence for non-blank tokens according to self.timestep

        Returns:
            List with confidence scores. The length of the list is the same as `timestep`.
        """
        non_blank_frame_confidence = []
        # self.timestep can be a dict for RNNT
        timestep = self.timestep['timestep'] if isinstance(self.timestep, dict) else self.timestep
        if len(self.timestep) != 0 and self.frame_confidence is not None:
            if any(isinstance(i, list) for i in self.frame_confidence):  # rnnt
                t_prev = -1
                offset = 0
                for t in timestep:
                    if t != t_prev:
                        t_prev = t
                        offset = 0
                    else:
                        offset += 1
                    non_blank_frame_confidence.append(self.frame_confidence[t][offset])
            else:  # ctc
                non_blank_frame_confidence = [self.frame_confidence[t] for t in timestep]
        return non_blank_frame_confidence

    @property
    def words(self) -> List[str]:
        """Get words from self.text

        Returns:
            List with words (str).
        """
        return [] if self.text is None else self.text.split()

def label_collate(labels, device=None):
    """Collates the label inputs for the rnn-t prediction network.
    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.
        device: Optional torch device to place the label on.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")

    batch_size = len(labels)
    max_len = max(len(label) for label in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    labels = torch.tensor(cat_labels, dtype=torch.int64, device=device)

    return labels

class RNNTDecoder(nn.Module):
    """A Recurrent Neural Network Transducer Decoder / Prediction Network (RNN-T Prediction Network).
    An RNN-T Decoder/Prediction network, comprised of a stateful LSTM model.

    Args:
        prednet: A dict-like object which contains the following key-value pairs.

            pred_hidden:
                int specifying the hidden dimension of the prediction net.

            pred_rnn_layers:
                int specifying the number of rnn layers.

            Optionally, it may also contain the following:

                forget_gate_bias:
                    float, set by default to 1.0, which constructs a forget gate
                    initialized to 1.0.
                    Reference:
                    [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

                t_max:
                    int value, set to None by default. If an int is specified, performs Chrono Initialization
                    of the LSTM network, based on the maximum number of timesteps `t_max` expected during the course
                    of training.
                    Reference:
                    [Can recurrent neural networks warp time?](https://openreview.net/forum?id=SJcKhk-Ab)

                weights_init_scale:
                    Float scale of the weights after initialization. Setting to lower than one
                    sometimes helps reduce variance between runs.

                hidden_hidden_bias_scale:
                    Float scale for the hidden-to-hidden bias scale. Set to 0.0 for
                    the default behaviour.

                dropout:
                    float, set to 0.0 by default. Optional dropout applied at the end of the final LSTM RNN layer.

        vocab_size: int, specifying the vocabulary size of the embedding layer of the Prediction network,
            excluding the RNNT blank token.

        normalization_mode: Can be either None, 'batch' or 'layer'. By default, is set to None.
            Defines the type of normalization applied to the RNN layer.

        random_state_sampling: bool, set to False by default. When set, provides normal-distribution
            sampled state tensors instead of zero tensors during training.
            Reference:
            [Recognizing long-form speech using streaming end-to-end models](https://arxiv.org/abs/1910.11455)

        blank_as_pad: bool, set to True by default. When set, will add a token to the Embedding layer of this
            prediction network, and will treat this token as a pad token. In essence, the RNNT pad token will
            be treated as a pad token, and the embedding layer will return a zero tensor for this token.

            It is set by default as it enables various batch optimizations required for batched beam search.
            Therefore, it is not recommended to disable this flag.
    """

    def __init__(
        self,
        prednet: Dict[str, Any],
        vocab_size: int,
        normalization_mode: Optional[str] = None,
        random_state_sampling: bool = False,
        blank_as_pad: bool = True,
        multisoftmax=False, #CTEMO
        language_masks=None, #CTEMO
        
    ):
        # Required arguments
        self.pred_hidden = prednet['pred_hidden']
        self.pred_rnn_layers = prednet["pred_rnn_layers"]
        self.blank_idx = vocab_size
        self.blank_as_pad = blank_as_pad
        # Initialize the model (blank token increases vocab size by 1)
        super().__init__()

        # Optional arguments
        forget_gate_bias = prednet.get('forget_gate_bias', 1.0)
        t_max = prednet.get('t_max', None)
        weights_init_scale = prednet.get('weights_init_scale', 1.0)
        hidden_hidden_bias_scale = prednet.get('hidden_hidden_bias_scale', 0.0)
        dropout = prednet.get('dropout', 0.0)
        self.random_state_sampling = random_state_sampling
        rnn_hidden_size = prednet.get("rnn_hidden_size", -1)
        
        self.prediction = nn.ModuleDict(
                    {
                        "embed": nn.Embedding(vocab_size + 1, self.pred_hidden, padding_idx=self.blank_idx),
                        "dec_rnn": LSTMDropout(
                            input_size=self.pred_hidden,
                            hidden_size=rnn_hidden_size if rnn_hidden_size > 0 else self.pred_hidden,
                            num_layers=self.pred_rnn_layers,
                            forget_gate_bias=forget_gate_bias,
                            t_max=t_max,
                            dropout=dropout,
                            weights_init_scale=weights_init_scale,
                            hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                            proj_size=self.pred_hidden if self.pred_hidden < rnn_hidden_size else 0,),})
        self._rnnt_export = False

        self.multisoftmax = multisoftmax #CTEMO
        self.language_masks = language_masks #CTEMO

    def forward(self, targets, target_length, states=None):
        # y: (B, U)
        y = label_collate(targets)

        # state maintenance is unnecessary during training forward call
        # to get state, use .predict() method.
        if self._rnnt_export:
            add_sos = False
        else:
            add_sos = True

        g, states = self.predict(y, state=states, add_sos=add_sos)  # (B, U, D)
        g = g.transpose(1, 2)  # (B, D, U)

        return g, target_length, states

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Stateful prediction of scores and state for a (possibly null) tokenset.
        This method takes various cases into consideration :
        - No token, no state - used for priming the RNN
        - No token, state provided - used for blank token scoring
        - Given token, states - used for scores + new states

        Here:
        B - batch size
        U - label length
        H - Hidden dimension size of RNN
        L - Number of RNN layers

        Args:
            y: Optional torch tensor of shape [B, U] of dtype long which will be passed to the Embedding.
                If None, creates a zero tensor of shape [B, 1, H] which mimics output of pad-token on EmbeddiNg.

            state: An optional list of states for the RNN. Eg: For LSTM, it is the state list length is 2.
                Each state must be a tensor of shape [L, B, H].
                If None, and during training mode and `random_state_sampling` is set, will sample a
                normal distribution tensor of the above shape. Otherwise, None will be passed to the RNN.

            add_sos: bool flag, whether a zero vector describing a "start of signal" token should be
                prepended to the above "y" tensor. When set, output size is (B, U + 1, H).

            batch_size: An optional int, specifying the batch size of the `y` tensor.
                Can be infered if `y` and `state` is None. But if both are None, then batch_size cannot be None.

        Returns:
            A tuple  (g, hid) such that -

            If add_sos is False:

                g:
                    (B, U, H)

                hid:
                    (h, c) where h is the final sequence hidden state and c is the final cell state:

                        h (tensor), shape (L, B, H)

                        c (tensor), shape (L, B, H)

            If add_sos is True:
                g:
                    (B, U + 1, H)

                hid:
                    (h, c) where h is the final sequence hidden state and c is the final cell state:

                        h (tensor), shape (L, B, H)

                        c (tensor), shape (L, B, H)

        """
        # Get device and dtype of current module
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            if y.device != device:
                y = y.to(device)

            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            # Y is not provided, assume zero tensor with shape [B, 1, H] is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size

            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

        # Prepend blank "start of sequence" symbol (zero tensor)
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        # If in training mode, and random_state_sampling is set,
        # initialize state to random normal distribution tensor.
        if state is None:
            if self.random_state_sampling and self.training:
                state = self.initialize_state(y)

        # Forward step through RNN
        y = y.transpose(0, 1)  # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)  # (B, U + 1, H)

        del y, start, state

        # Adapter module forward step
        # if self.is_adapter_available():
        #     g = self.forward_enabled_adapters(g)

        return g, hid

    def initialize_state(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the state of the LSTM layers, with same dtype and device as input `y`.
        LSTM accepts a tuple of 2 tensors as a state.

        Args:
            y: A torch.Tensor whose device the generated states will be placed on.

        Returns:
            Tuple of 2 tensors, each of shape [L, B, H], where

                L = Number of RNN layers

                B = Batch size

                H = Hidden size of RNN.
        """
        batch = y.size(0)
        if self.random_state_sampling and self.training:
            state = (
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.randn(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            )

        else:
            state = (
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
                torch.zeros(self.pred_rnn_layers, batch, self.pred_hidden, dtype=y.dtype, device=y.device),
            )
        return state

    def batch_select_state(self, batch_states: List[torch.Tensor], idx: int) -> List[List[torch.Tensor]]:
        """Get decoder state from batch of states, for given id.

        Args:
            batch_states (list): batch of decoder states
                ([L x (B, H)], [L x (B, H)])

            idx (int): index to extract state from batch of states

        Returns:
            (tuple): decoder states for given id
                ([L x (1, H)], [L x (1, H)])
        """
        if batch_states is not None:
            state_list = []
            for state_id in range(len(batch_states)):
                states = [batch_states[state_id][layer][idx] for layer in range(self.pred_rnn_layers)]
                state_list.append(states)

            return state_list
        else:
            return None

    def batch_copy_states(
        self,
        old_states: List[torch.Tensor],
        new_states: List[torch.Tensor],
        ids: List[int],
        value: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """Copy states from new state to old state at certain indices.

        Args:
            old_states(list): packed decoder states
                (L x B x H, L x B x H)

            new_states: packed decoder states
                (L x B x H, L x B x H)

            ids (list): List of indices to copy states at.

            value (optional float): If a value should be copied instead of a state slice, a float should be provided

        Returns:
            batch of decoder states with partial copy at ids (or a specific value).
                (L x B x H, L x B x H)
        """
        for state_id in range(len(old_states)):
            if value is None:
                old_states[state_id][:, ids, :] = new_states[state_id][:, ids, :]
            else:
                old_states[state_id][:, ids, :] *= 0.0
                old_states[state_id][:, ids, :] += value

        return old_states

class RNNTJoint(nn.Module):
    """A Recurrent Neural Network Transducer Joint Network (RNN-T Joint Network).
    An RNN-T Joint network, comprised of a feedforward model.

    Args:
        jointnet: A dict-like object which contains the following key-value pairs.
            encoder_hidden: int specifying the hidden dimension of the encoder net.
            pred_hidden: int specifying the hidden dimension of the prediction net.
            joint_hidden: int specifying the hidden dimension of the joint net
            activation: Activation function used in the joint step. Can be one of
            ['relu', 'tanh', 'sigmoid'].

            Optionally, it may also contain the following:
            dropout: float, set to 0.0 by default. Optional dropout applied at the end of the joint net.

        num_classes: int, specifying the vocabulary size that the joint network must predict,
            excluding the RNNT blank token.

        vocabulary: Optional list of strings/tokens that comprise the vocabulary of the joint network.
            Unused and kept only for easy access for character based encoding RNNT models.

        log_softmax: Optional bool, set to None by default. If set as None, will compute the log_softmax()
            based on the value provided.

        preserve_memory: Optional bool, set to False by default. If the model crashes due to the memory
            intensive joint step, one might try this flag to empty the tensor cache in pytorch.

            Warning: This will make the forward-backward pass much slower than normal.
            It also might not fix the OOM if the GPU simply does not have enough memory to compute the joint.

        fuse_loss_wer: Optional bool, set to False by default.

            Fuses the joint forward, loss forward and
            wer forward steps. In doing so, it trades of speed for memory conservation by creating sub-batches
            of the provided batch of inputs, and performs Joint forward, loss forward and wer forward (optional),
            all on sub-batches, then collates results to be exactly equal to results from the entire batch.

            When this flag is set, prior to calling forward, the fields `loss` and `wer` (either one) *must*
            be set using the `RNNTJoint.set_loss()` or `RNNTJoint.set_wer()` methods.

            Further, when this flag is set, the following argument `fused_batch_size` *must* be provided
            as a non negative integer. This value refers to the size of the sub-batch.

            When the flag is set, the input and output signature of `forward()` of this method changes.
            Input - in addition to `encoder_outputs` (mandatory argument), the following arguments can be provided.

                - decoder_outputs (optional). Required if loss computation is required.

                - encoder_lengths (required)

                - transcripts (optional). Required for wer calculation.

                - transcript_lengths (optional). Required for wer calculation.

                - compute_wer (bool, default false). Whether to compute WER or not for the fused batch.

            Output - instead of the usual `joint` log prob tensor, the following results can be returned.

                - loss (optional). Returned if decoder_outputs, transcripts and transript_lengths are not None.

                - wer_numerator + wer_denominator (optional). Returned if transcripts, transcripts_lengths are provided
                    and compute_wer is set.

        fused_batch_size: Optional int, required if `fuse_loss_wer` flag is set. Determines the size of the
            sub-batches. Should be any value below the actual batch size per GPU.
    """

    def __init__(
        self,
        jointnet: Dict[str, Any],
        num_classes: int,
        num_extra_outputs: int = 0,
        vocabulary: Optional[List] = None,
        log_softmax: Optional[bool] = None,
        preserve_memory: bool = False,
        fuse_loss_wer: bool = False,
        fused_batch_size: Optional[int] = None,
        experimental_fuse_loss_wer: Any = None,
        language_masks=None, #CTEMO
        multilingual: bool = False, #CTEMO
        language_keys: Optional[List] = None, #CTEMO
        token_id_offsets=None, #CTEMO
        offset_token_ids_by_token_id=None, #CTEMO
    ):
        super().__init__()

        self.vocabulary = vocabulary

        self._vocab_size = num_classes
        self._num_extra_outputs = num_extra_outputs
        self._num_classes = num_classes + 1 + num_extra_outputs  # 1 is for blank
        self.language_masks = language_masks #CTEMO
        self.token_id_offsets = token_id_offsets #CTEMO
        self.offset_token_ids_by_token_id = offset_token_ids_by_token_id #CTEMO
        self.multilingual = multilingual #CTEMO
        self.language_keys = language_keys #CTEMO

        if experimental_fuse_loss_wer is not None:
            # Override fuse_loss_wer from deprecated argument
            fuse_loss_wer = experimental_fuse_loss_wer

        self._fuse_loss_wer = fuse_loss_wer
        self._fused_batch_size = fused_batch_size

        if fuse_loss_wer and (fused_batch_size is None):
            raise ValueError("If `fuse_loss_wer` is set, then `fused_batch_size` cannot be None!")

        self._loss = None
        self._wer = None

        # Log softmax should be applied explicitly only for CPU
        self.log_softmax = log_softmax
        self.preserve_memory = preserve_memory

        if preserve_memory:
            logging.warning(
                "`preserve_memory` was set for the Joint Model. Please be aware this will severely impact "
                "the forward-backward step time. It also might not solve OOM issues if the GPU simply "
                "does not have enough memory to compute the joint."
            )

        # Required arguments
        self.encoder_hidden = jointnet['encoder_hidden']
        self.pred_hidden = jointnet['pred_hidden']
        self.joint_hidden = jointnet['joint_hidden']
        self.activation = jointnet['activation']

        # Optional arguments
        dropout = jointnet.get('dropout', 0.0)

        self.pred, self.enc, self.joint_net = self._joint_net_modules(
            num_classes=self._num_classes,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            enc_n_hidden=self.encoder_hidden,
            joint_n_hidden=self.joint_hidden,
            activation=self.activation,
            dropout=dropout,
        )

        # Flag needed for RNNT export support
        self._rnnt_export = False

        # to change, requires running ``model.temperature = T`` explicitly
        self.temperature = 1.0

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_outputs: Optional[torch.Tensor],
        encoder_lengths: Optional[torch.Tensor] = None,
        transcripts: Optional[torch.Tensor] = None,
        transcript_lengths: Optional[torch.Tensor] = None,
        compute_wer: bool = False,
        language_ids=None, #CTEMO
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # encoder = (B, D, T)
        # decoder = (B, D, U) if passed, else None
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)

        if decoder_outputs is not None:
            decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        if not self._fuse_loss_wer:
            if decoder_outputs is None:
                raise ValueError(
                    "decoder_outputs passed is None, and `fuse_loss_wer` is not set. "
                    "decoder_outputs can only be None for fused step!"
                )

            out = self.joint(encoder_outputs, decoder_outputs, language_ids=language_ids)  # [B, T, U, V + 1] #CTEMO
            return out

        else:
            # At least the loss module must be supplied during fused joint
            if self._loss is None or self._wer is None:
                raise ValueError("`fuse_loss_wer` flag is set, but `loss` and `wer` modules were not provided! ")

            # If fused joint step is required, fused batch size is required as well
            if self._fused_batch_size is None:
                raise ValueError("If `fuse_loss_wer` is set, then `fused_batch_size` cannot be None!")

            # When using fused joint step, both encoder and transcript lengths must be provided
            if (encoder_lengths is None) or (transcript_lengths is None):
                raise ValueError(
                    "`fuse_loss_wer` is set, therefore encoder and target lengths " "must be provided as well!"
                )

            losses = []
            wers, wer_nums, wer_denoms = [], [], []
            target_lengths = []
            batch_size = int(encoder_outputs.size(0))  # actual batch size

            # Iterate over batch using fused_batch_size steps
            for batch_idx in range(0, batch_size, self._fused_batch_size):
                begin = batch_idx
                end = min(begin + self._fused_batch_size, batch_size)

                # Extract the sub batch inputs
                # sub_enc = encoder_outputs[begin:end, ...]
                # sub_transcripts = transcripts[begin:end, ...]
                sub_enc = encoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))
                sub_transcripts = transcripts.narrow(dim=0, start=begin, length=int(end - begin))

                sub_enc_lens = encoder_lengths[begin:end]
                sub_transcript_lens = transcript_lengths[begin:end]

                # Sub transcripts does not need the full padding of the entire batch
                # Therefore reduce the decoder time steps to match
                max_sub_enc_length = sub_enc_lens.max()
                max_sub_transcript_length = sub_transcript_lens.max()

                if decoder_outputs is not None:
                    # Reduce encoder length to preserve computation
                    # Encoder: [sub-batch, T, D] -> [sub-batch, T', D]; T' < T
                    if sub_enc.shape[1] != max_sub_enc_length:
                        sub_enc = sub_enc.narrow(dim=1, start=0, length=int(max_sub_enc_length))

                    # sub_dec = decoder_outputs[begin:end, ...]  # [sub-batch, U, D]
                    sub_dec = decoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))  # [sub-batch, U, D]

                    # Reduce decoder length to preserve computation
                    # Decoder: [sub-batch, U, D] -> [sub-batch, U', D]; U' < U
                    if sub_dec.shape[1] != max_sub_transcript_length + 1:
                        sub_dec = sub_dec.narrow(dim=1, start=0, length=int(max_sub_transcript_length + 1))

                    # Perform joint => [sub-batch, T', U', V + 1]
                    if language_ids is not None: #CTEMO
                        sub_joint = self.joint(sub_enc, sub_dec, language_ids=language_ids[begin:end]) #CTEMO
                    else:
                        sub_joint = self.joint(sub_enc, sub_dec) #CTEMO

                    del sub_dec

                    # Reduce transcript length to correct alignment
                    # Transcript: [sub-batch, L] -> [sub-batch, L']; L' <= L
                    if sub_transcripts.shape[1] != max_sub_transcript_length:
                        sub_transcripts = sub_transcripts.narrow(dim=1, start=0, length=int(max_sub_transcript_length))

                    # Compute sub batch loss
                    # preserve loss reduction type
                    loss_reduction = self.loss.reduction

                    # override loss reduction to sum
                    self.loss.reduction = None

                    # compute and preserve loss
                    loss_batch = self.loss(
                        log_probs=sub_joint,
                        targets=sub_transcripts,
                        input_lengths=sub_enc_lens,
                        target_lengths=sub_transcript_lens,
                    )
                    losses.append(loss_batch)
                    target_lengths.append(sub_transcript_lens)

                    # reset loss reduction type
                    self.loss.reduction = loss_reduction

                else:
                    losses = None

                # Update WER for sub batch
                if compute_wer:
                    sub_enc = sub_enc.transpose(1, 2)  # [B, T, D] -> [B, D, T]
                    sub_enc = sub_enc.detach()
                    sub_transcripts = sub_transcripts.detach()

                    # Update WER on each process without syncing
                    if language_ids is not None: #CTEMO
                        self.wer.update(
                            predictions=sub_enc,
                            predictions_lengths=sub_enc_lens,
                            targets=sub_transcripts,
                            targets_lengths=sub_transcript_lens,
                            lang_ids=language_ids[begin:end]
                        )
                    else:
                        self.wer.update(
                            predictions=sub_enc,
                            predictions_lengths=sub_enc_lens,
                            targets=sub_transcripts,
                            targets_lengths=sub_transcript_lens,
                        )
                    # Sync and all_reduce on all processes, compute global WER
                    wer, wer_num, wer_denom = self.wer.compute()
                    self.wer.reset()

                    wers.append(wer)
                    wer_nums.append(wer_num)
                    wer_denoms.append(wer_denom)

                del sub_enc, sub_transcripts, sub_enc_lens, sub_transcript_lens

            # Reduce over sub batches
            if losses is not None:
                losses = self.loss.reduce(losses, target_lengths)

            # Collect sub batch wer results
            if compute_wer:
                wer = sum(wers) / len(wers)
                wer_num = sum(wer_nums)
                wer_denom = sum(wer_denoms)
            else:
                wer = None
                wer_num = None
                wer_denom = None

            return losses, wer, wer_num, wer_denom

    def joint(self, f: torch.Tensor, g: torch.Tensor, language_ids=None) -> torch.Tensor:
        return self.joint_after_projection(self.project_encoder(f), self.project_prednet(g), language_ids)

    def project_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Project the encoder output to the joint hidden dimension.

        Args:
            encoder_output: A torch.Tensor of shape [B, T, D]

        Returns:
            A torch.Tensor of shape [B, T, H]
        """
        return self.enc(encoder_output)

    def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
        """
        Project the Prediction Network (Decoder) output to the joint hidden dimension.

        Args:
            prednet_output: A torch.Tensor of shape [B, U, D]

        Returns:
            A torch.Tensor of shape [B, U, H]
        """
        return self.pred(prednet_output)

    def joint_after_projection(self, f: torch.Tensor, g: torch.Tensor, language_ids=None) -> torch.Tensor: #CTEMO
        """
        Compute the joint step of the network after projection.

        Here,
        B = Batch size
        T = Acoustic model timesteps
        U = Target sequence length
        H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
        H = Hidden dimension of the Joint hidden step.
        V = Vocabulary size of the Decoder (excluding the RNNT blank token).

        NOTE:
            The implementation of this model is slightly modified from the original paper.
            The original paper proposes the following steps :
            (enc, dec) -> Expand + Concat + Sum [B, T, U, H1+H2] -> Forward through joint hidden [B, T, U, H] -- *1
            *1 -> Forward through joint final [B, T, U, V + 1].

            We instead split the joint hidden into joint_hidden_enc and joint_hidden_dec and act as follows:
            enc -> Forward through joint_hidden_enc -> Expand [B, T, 1, H] -- *1
            dec -> Forward through joint_hidden_dec -> Expand [B, 1, U, H] -- *2
            (*1, *2) -> Sum [B, T, U, H] -> Forward through joint final [B, T, U, V + 1].

        Args:
            f: Output of the Encoder model. A torch.Tensor of shape [B, T, H1]
            g: Output of the Decoder model. A torch.Tensor of shape [B, U, H2]

        Returns:
            Logits / log softmaxed tensor of shape (B, T, U, V + 1).
        """
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)
        g = g.unsqueeze(dim=1)  # (B, 1, U, H)
        inp = f + g  # [B, T, U, H]

        del f, g

        # Forward adapter modules on joint hidden
        # if self.is_adapter_available():
        #     inp = self.forward_enabled_adapters(inp)

        if language_ids is not None: #CTEMO
            # Do partial forward of joint net (skipping the final linear)
            for module in self.joint_net[:-1]:
                inp = module(inp)  # [B, T, U, H]
            
            # check if all the items in the batch have the same langauge, pass them through
            if len(set(language_ids)) == 1:
                res = self.joint_net[-1][language_ids[0]](inp)
            else:
                res_single = []
                for single_inp, lang in zip(inp, language_ids):
                    res_single.append(self.joint_net[-1][lang](single_inp))
                res = torch.stack(res_single)
        else:
            res = self.joint_net(inp)  # [B, T, U, V + 1]

        del inp

        if self.preserve_memory:
            torch.cuda.empty_cache()

        # If log_softmax is automatic
        if self.log_softmax is None:
            if not res.is_cuda:  # Use log softmax only if on CPU
                if self.temperature != 1.0:
                    res = (res / self.temperature).log_softmax(dim=-1)
                else:
                    res = res.log_softmax(dim=-1)
        else:
            if self.log_softmax:
                if self.temperature != 1.0:
                    res = (res / self.temperature).log_softmax(dim=-1)
                else:
                    res = res.log_softmax(dim=-1)

        return res

    def _joint_net_modules(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout):
        """
        Prepare the trainable modules of the Joint Network

        Args:
            num_classes: Number of output classes (vocab size) excluding the RNNT blank token.
            pred_n_hidden: Hidden size of the prediction network.
            enc_n_hidden: Hidden size of the encoder network.
            joint_n_hidden: Hidden size of the joint network.
            activation: Activation of the joint. Can be one of [relu, tanh, sigmoid]
            dropout: Dropout value to apply to joint.
        """
        pred = nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = nn.Linear(enc_n_hidden, joint_n_hidden)

        if activation not in ['relu', 'sigmoid', 'tanh']:
            raise ValueError("Unsupported activation for joint step - please pass one of " "[relu, sigmoid, tanh]")

        activation = activation.lower()

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'tanh':
            activation = nn.Tanh()

        if self.multilingual: #CTEMO
            final_layer = nn.ModuleDict()
            logging.info(f"Vocab size for each language: {self._vocab_size // len(self.language_keys)}")
            for lang in self.language_keys:
                final_layer[lang] = nn.Linear(joint_n_hidden, (self._vocab_size // len(self.language_keys)+1))
            layers = (
                [activation]
                + ([nn.Dropout(p=dropout)] if dropout else [])
                + [final_layer]
            )
        else:
            layers = (
                [activation]
                + ([nn.Dropout(p=dropout)] if dropout else [])
                + [nn.Linear(joint_n_hidden, num_classes)]
            )
        return pred, enc, nn.Sequential(*layers)

    @property
    def num_classes_with_blank(self):
        return self._num_classes

    @property
    def num_extra_outputs(self):
        return self._num_extra_outputs

    @property
    def loss(self):
        return self._loss

    def set_loss(self, loss):
        if not self._fuse_loss_wer:
            raise ValueError("Attempting to set loss module even though `fuse_loss_wer` is not set!")

        self._loss = loss

    @property
    def wer(self):
        return self._wer

    def set_wer(self, wer):
        if not self._fuse_loss_wer:
            raise ValueError("Attempting to set WER module even though `fuse_loss_wer` is not set!")

        self._wer = wer

    @property
    def fuse_loss_wer(self):
        return self._fuse_loss_wer

    @property
    def fused_batch_size(self):
        return self._fused_batch_size

class GreedyRNNTInfer:

    def __init__(
        self,
        decoder_model: RNNTDecoder,
        joint_model: RNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
        preserve_frame_confidence: bool = False,
    ):
        super().__init__()
        self.decoder = decoder_model
        self.joint = joint_model

        self._blank_index = blank_index
        self._SOS = blank_index  # Start of single index

        if max_symbols_per_step is not None and max_symbols_per_step <= 0:
            raise ValueError(f"Expected max_symbols_per_step > 0 (or None), got {max_symbols_per_step}")
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        hidden: Optional[torch.Tensor],
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Common prediction step based on the AbstractRNNTDecoder implementation.

        Args:
            label: (int/torch.Tensor): Label or "Start-of-Signal" token.
            hidden: (Optional torch.Tensor): RNN State vector
            add_sos (bool): Whether to add a zero vector at the begging as "start of sentence" token.
            batch_size: Batch size of the output tensor.

        Returns:
            g: (B, U, H) if add_sos is false, else (B, U + 1, H)
            hid: (h, c) where h is the final sequence hidden state and c is
                the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)
        """
        if isinstance(label, torch.Tensor):
            # label: [batch, 1]
            if label.dtype != torch.long:
                label = label.long()

        else:
            # Label is an integer
            if label == self._SOS:
                return self.decoder.predict(None, hidden, add_sos=add_sos, batch_size=batch_size)

            label = label_collate([[label]])

        # output: [B, 1, K]
        return self.decoder.predict(label, hidden, add_sos=add_sos, batch_size=batch_size)

    def _joint_step(self, enc, pred, log_normalize: Optional[bool] = None, language_ids=None): # CTEMO
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model. A torch.Tensor of shape [B, 1, H1]
            pred: Output of the Decoder model. A torch.Tensor of shape [B, 1, H2]
            log_normalize: Whether to log normalize or not. None will log normalize only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        with torch.no_grad():
            #Old
            # # logits = self.joint.joint(enc, pred)
            ## New for multisoftmax
            self.joint._fuse_loss_wer = False
            logits = self.joint(encoder_outputs=enc.transpose(1, 2), decoder_outputs=pred.transpose(1, 2), language_ids=language_ids)
            self.joint._fuse_loss_wer = True
            if log_normalize is None:
                if not logits.is_cuda:  # Use log softmax only if on CPU
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)
            else:
                if log_normalize:
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits

    def _greedy_decode_blank_as_pad_loop_frames(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[List[Hypothesis]] = None,
        language_ids=None,
    ):
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        with torch.inference_mode():
            # x: [B, T, D]
            # out_len: [B]
            # device: torch.device

            # Initialize list of Hypothesis
            batchsize = x.shape[0]
            hypotheses = [
                Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)
            ]

            # Initialize Hidden state matrix (shared by entire batch)
            hidden = None

            # If alignments need to be preserved, register a dangling list to hold the values
            if self.preserve_alignments:
                # alignments is a 3-dimensional dangling list representing B x T x U
                for hyp in hypotheses:
                    hyp.alignments = [[]]

            # If confidence scores need to be preserved, register a dangling list to hold the values
            if self.preserve_frame_confidence:
                # frame_confidence is a 3-dimensional dangling list representing B x T x U
                for hyp in hypotheses:
                    hyp.frame_confidence = [[]]

            # Last Label buffer + Last Label without blank buffer
            # batch level equivalent of the last_label
            last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long, device=device)

            # Mask buffers
            blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)
            blank_mask_prev = None

            # Get max sequence length
            max_out_len = out_len.max()
            for time_idx in range(max_out_len):
                f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

                # Prepare t timestamp batch variables
                not_blank = True
                symbols_added = 0

                # Reset blank mask
                blank_mask.mul_(False)

                # Update blank mask with time mask
                # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
                # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
                blank_mask = time_idx >= out_len
                blank_mask_prev = blank_mask.clone()

                # Start inner loop
                while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                    # Batch prediction and joint network steps
                    # If very first prediction step, submit SOS tag (blank) to pred_step.
                    # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                    if time_idx == 0 and symbols_added == 0 and hidden is None:
                        g, hidden_prime = self._pred_step(self._SOS, hidden, batch_size=batchsize)
                    else:
                        # Perform batch step prediction of decoder, getting new states and scores ("g")
                        g, hidden_prime = self._pred_step(last_label, hidden, batch_size=batchsize)

                    # Batched joint step - Output = [B, V + 1]
                    # If preserving per-frame confidence, log_normalize must be true
                    logp = self._joint_step(f, g, log_normalize=True if self.preserve_frame_confidence else None, language_ids=language_ids)[
                        :, 0, 0, :
                    ]

                    if logp.dtype != torch.float32:
                        logp = logp.float()

                    # Get index k, of max prob for batch
                    v, k = logp.max(1)
                    del g

                    # Update blank mask with current predicted blanks
                    # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                    k_is_blank = k == self._blank_index
                    blank_mask.bitwise_or_(k_is_blank)

                    del k_is_blank

                    # If preserving alignments, check if sequence length of sample has been reached
                    # before adding alignment
                    if self.preserve_alignments:
                        # Insert logprobs into last timestep per sample
                        logp_vals = logp.to('cpu')
                        logp_ids = logp_vals.max(1)[1]
                        for batch_idx, is_blank in enumerate(blank_mask):
                            # we only want to update non-blanks and first-time blanks,
                            # otherwise alignments will contain duplicate predictions
                            if time_idx < out_len[batch_idx] and (not blank_mask_prev[batch_idx] or not is_blank):
                                hypotheses[batch_idx].alignments[-1].append(
                                    (logp_vals[batch_idx], logp_ids[batch_idx])
                                )
                        del logp_vals

                    # If preserving per-frame confidence, check if sequence length of sample has been reached
                    # before adding confidence scores
                    if self.preserve_frame_confidence:
                        # Insert probabilities into last timestep per sample
                        confidence = self._get_confidence(logp)
                        for batch_idx, is_blank in enumerate(blank_mask):
                            if time_idx < out_len[batch_idx] and (not blank_mask_prev[batch_idx] or not is_blank):
                                hypotheses[batch_idx].frame_confidence[-1].append(confidence[batch_idx])
                    del logp

                    blank_mask_prev.bitwise_or_(blank_mask)

                    # If all samples predict / have predicted prior blanks, exit loop early
                    # This is equivalent to if single sample predicted k
                    if blank_mask.all():
                        not_blank = False
                    else:
                        # Collect batch indices where blanks occurred now/past
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)

                        # Recover prior state for all samples which predicted blank now/past
                        if hidden is not None:
                            # LSTM has 2 states
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, hidden, blank_indices)

                        elif len(blank_indices) > 0 and hidden is None:
                            # Reset state if there were some blank and other non-blank predictions in batch
                            # Original state is filled with zeros so we just multiply
                            # LSTM has 2 states
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, None, blank_indices, value=0.0)

                        # Recover prior predicted label for all samples which predicted blank now/past
                        k[blank_indices] = last_label[blank_indices, 0]

                        # Update new label and hidden state for next iteration
                        last_label = k.clone().view(-1, 1)
                        hidden = hidden_prime

                        # Update predicted labels, accounting for time mask
                        # If blank was predicted even once, now or in the past,
                        # Force the current predicted label to also be blank
                        # This ensures that blanks propogate across all timesteps
                        # once they have occured (normally stopping condition of sample level loop).
                        for kidx, ki in enumerate(k):
                            if blank_mask[kidx] == 0:
                                hypotheses[kidx].y_sequence.append(ki)
                                hypotheses[kidx].timestep.append(time_idx)
                                hypotheses[kidx].score += float(v[kidx])
                        symbols_added += 1

                # If preserving alignments, convert the current Uj alignments into a torch.Tensor
                # Then preserve U at current timestep Ti
                # Finally, forward the timestep history to Ti+1 for that sample
                # All of this should only be done iff the current time index <= sample-level AM length.
                # Otherwise ignore and move to next sample / next timestep.
                if self.preserve_alignments:

                    # convert Ti-th logits into a torch array
                    for batch_idx in range(batchsize):

                        # this checks if current timestep <= sample-level AM length
                        # If current timestep > sample-level AM length, no alignments will be added
                        # Therefore the list of Uj alignments is empty here.
                        if len(hypotheses[batch_idx].alignments[-1]) > 0:
                            hypotheses[batch_idx].alignments.append([])  # blank buffer for next timestep

                # Do the same if preserving per-frame confidence
                if self.preserve_frame_confidence:

                    for batch_idx in range(batchsize):
                        if len(hypotheses[batch_idx].frame_confidence[-1]) > 0:
                            hypotheses[batch_idx].frame_confidence.append([])  # blank buffer for next timestep

            # Remove trailing empty list of alignments at T_{am-len} x Uj
            if self.preserve_alignments:
                for batch_idx in range(batchsize):
                    if len(hypotheses[batch_idx].alignments[-1]) == 0:
                        del hypotheses[batch_idx].alignments[-1]

            # Remove trailing empty list of confidence scores at T_{am-len} x Uj
            if self.preserve_frame_confidence:
                for batch_idx in range(batchsize):
                    if len(hypotheses[batch_idx].frame_confidence[-1]) == 0:
                        del hypotheses[batch_idx].frame_confidence[-1]

        # Preserve states
        for batch_idx in range(batchsize):
            hypotheses[batch_idx].dec_state = self.decoder.batch_select_state(hidden, batch_idx)

        return hypotheses
