
import json
import math
import re
from typing import List
import torch
from torch import nn
from tqdm import tqdm
from shruti.nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
from shruti.nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInfer
from shruti.utils import ConformerLayer, ConvSubsampling, Hypothesis, MelPreprocessor, RelPositionalEncoding, create_masks
import torchaudio
blank_id = 256

class ASR(nn.Module):
    def __init__(self):
        super().__init__()
        n_layers = 24
        self.preprocessor = MelPreprocessor()
        self.pre_encode = ConvSubsampling(
            subsampling='dw_striding',
            subsampling_factor=8,
            feat_in=80,
            feat_out=1024,
            conv_channels=256,
            activation=nn.ReLU(True),
        )
        self.pos_enc = RelPositionalEncoding(1024,xscale=math.sqrt(1024)) # dropout remove for inference
        pos_bias_u = nn.Parameter(torch.Tensor(8, 1024 // 8))
        pos_bias_v = nn.Parameter(torch.Tensor(8, 1024 // 8))
        nn.init.zeros_(pos_bias_u)
        nn.init.zeros_(pos_bias_v)
        self.layers = nn.ModuleList([ConformerLayer(
                d_model=1024,
                d_ff=1024 * 4,
                n_heads=8,
                conv_kernel_size=9,
                conv_context_size=[4,4],
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
            ) for _ in n_layers])
        self.decoder = RNNTDecoder({'pred_hidden': 640, 'pred_rnn_layers': 2, 't_max': None, 'dropout': 0.2},5632,multisoftmax=True)
        # TODO Add Tokenizer
        # self.tokenizer = T
        self.joint = RNNTJoint(
            {'joint_hidden': 640, 'activation': 'relu', 'dropout': 0.2, 'encoder_hidden': 1024, 'pred_hidden': 640},
            5632,vocabulary=json.load(open("vocab.json"))

        )

        # GET OTHER PARAMS FOR RNNTJoint
        # def get_hparams(module):
        #     hparams = {}
        #     for name, value in module.__dict__.items():
        #         if not name.startswith("_") and not callable(value):
        #             hparams[name] = value
        #     return hparams
        # c = {i:v for i,v in get_hparams(asr.model.joint).items() if not i == "vocabulary"}


        self.decoding = GreedyBatchedRNNTInfer(
                        decoder_model=self.decoder,
                        joint_model=self.joint,
                        blank_index=self.blank_id,
                        max_symbols_per_step=(
                            self.cfg.greedy.get('max_symbols', None)
                            or self.cfg.greedy.get('max_symbols_per_step', None)
                        ),
                        preserve_alignments=self.preserve_alignments,
                        preserve_frame_confidence=self.preserve_frame_confidence,
                        confidence_method_cfg=self.confidence_method_cfg,
                        loop_labels=self.cfg.greedy.get('loop_labels', False),
                        use_cuda_graph_decoder=self.cfg.greedy.get('use_cuda_graph_decoder', False),
                    )

    def forward(self):
        with torch.inference_mode():
            x,sr = torchaudio.load("audio.mp3")
            if x.shape[0] > 1:
                x = x.mean(dim=0)
            if sr != 16000:
                x = torchaudio.functional.resample(x, sr, 16000)
            x,l = self.preprocessor.forward(x.unsqueeze(0),torch.tensor([x.shape[-1]]))
            x,l = self.pre_encode.forward(x.transpose(1, 2),l)
            x , pos_emb = self.pos_enc.forward(x)
            l = l.to(torch.int64)
            pad_mask, att_mask = create_masks(
                        att_context_size=[-1, -1],
                        padding_length=l,
                        max_audio_length=x.size(1),
                        device=x.device,
            )
            for i in tqdm(self.layers):
                x = i.forward(x=x,att_mask=att_mask,pos_emb=pos_emb,pad_mask=pad_mask)
            h = self.decoding(encoder_output=x.transpose(1,2), encoded_lengths=l, partial_hypotheses=None, language_ids=["hi"] * len(x))
            h = self.decode_hypothesis(h,["hi"] * len(x))

    def decode_hypothesis(self,hypotheses_list: List[Hypothesis], lang_ids: List[str] = None) -> List[Hypothesis]:
        for ind in range(len(hypotheses_list)):
            prediction = hypotheses_list[ind].y_sequence

            if type(prediction) != list:
                prediction = prediction.tolist()
            prediction = [p for p in prediction if p != blank_id]

            if lang_ids is not None: #CTEMO
                hypothesis = self.tokenizer.ids_to_text(prediction, lang_ids[ind])
            else:
                hypothesis = self.tokenizer.ids_to_text(prediction)
                hypothesis = re.sub(r'(\s+)([\.\,\?])', r'\2', hypothesis)
            hypotheses_list[ind].text = hypothesis

        return hypotheses_list
