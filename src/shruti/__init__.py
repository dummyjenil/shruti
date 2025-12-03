import gc
import math
from pathlib import Path
from torch import nn
from tqdm import tqdm
from shruti.utils import BLANK_ID, ConformerLayer, ConvSubsampling, GreedyBatchedRNNTInfer, MelPreprocessor, MultilingualTokenizer, RNNTDecoder, RNNTJoint, RelPositionalEncoding, SentencePieceTokenizer, create_masks, decode_hypothesis, pack_hypotheses , ChunkedData , padding_audio , make_srt
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
import srt
from torch.utils.data import DataLoader
class ShrutiASR(nn.Module):
    def __init__(self):
        super().__init__()
        model_path = Path(snapshot_download("shethjenil/shruti"))
        d_model = 1024
        ff_expansion_factor = 4
        n_layers = 24
        n_heads = 8
        conv_kernel_size = 9
        pred_hidden = 640
        pred_rnn_layers = 2
        joint_hidden = 640
        max_symbols = 10
        conv_channels = 256
        subsampling_factor = 8
        feat_in = 80
        # TODO bhaai aa real preprocessor nathi FilterbankFeatures aa real chhe teni args joi levi
        self.preprocessor = MelPreprocessor()
        self.pre_encode = ConvSubsampling('dw_striding',subsampling_factor,feat_in,d_model,conv_channels,activation=nn.ReLU(True))
        self.pos_enc = RelPositionalEncoding(d_model,xscale=math.sqrt(d_model))
        self.pos_enc.extend_pe(5000)
        pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_model // n_heads))
        pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_model // n_heads))
        nn.init.zeros_(pos_bias_u)
        nn.init.zeros_(pos_bias_v)
        self.enc_layers = nn.ModuleList([ConformerLayer(d_model,d_model * ff_expansion_factor,n_heads,conv_kernel_size,conv_context_size=[4,4],pos_bias_u=pos_bias_u,pos_bias_v=pos_bias_v) for _ in range(n_layers)])
        self.tokenizer = MultilingualTokenizer({i.name:SentencePieceTokenizer(str((i/"tokenizer.model").absolute())) for i in (model_path/"tokenizer").rglob("*/")})
        self.decoder = RNNTDecoder({'pred_hidden': pred_hidden, 'pred_rnn_layers': pred_rnn_layers, 't_max': None, 'dropout': 0.2},self.tokenizer.vocab_size,multisoftmax=True)
        self.joint = RNNTJoint(
            {'joint_hidden': joint_hidden, 'activation': 'relu', 'dropout': 0.2, 'encoder_hidden': d_model, 'pred_hidden': pred_hidden},
            self.tokenizer.vocab_size,
            vocabulary=self.tokenizer.vocabulary,
            multilingual=True,
            token_id_offsets=self.tokenizer.token_id_offset,
            language_keys=self.tokenizer.langs,
            language_masks={lang: [(token_lang == lang) for _, token_lang in self.tokenizer.langs_by_token_id.items()] + [True] for lang in self.tokenizer.tokenizers_dict},
            offset_token_ids_by_token_id = self.tokenizer.offset_token_ids_by_token_id
            )
        self.greedy_decoder = GreedyBatchedRNNTInfer(self.decoder,self.joint,BLANK_ID,max_symbols,confidence_method_cfg={'name': 'entropy', 'entropy_type': 'tsallis', 'alpha': 0.33, 'entropy_norm': 'exp', 'temperature': 'DEPRECATED'},)
        self.load_state_dict(load_file(model_path/"model.safetensors"),strict=False)
    def encoder(self,audio_tensor,length_tensor):
        audio_tensor,length_tensor = self.preprocessor.forward(audio_tensor,length_tensor)
        audio_tensor,length_tensor = self.pre_encode.forward(audio_tensor.transpose(1, 2),length_tensor)
        length_tensor = length_tensor.to(torch.int64)
        audio_tensor , pos_emb = self.pos_enc.forward(audio_tensor)
        pad_mask, att_mask = create_masks(length_tensor,audio_tensor.size(1))
        for i in tqdm(self.enc_layers):
            audio_tensor = i.forward(audio_tensor,att_mask,pos_emb,pad_mask)
        return audio_tensor , length_tensor
    def decoding_with_lang(self,audio_tensor,length_tensor,language):
        l_id = [language] * len(audio_tensor)
        # return decode_hypothesis(self.tokenizer,pack_hypotheses(self.greedy_decoder._greedy_decode_blank_as_pad_loop_frames(audio_tensor, length_tensor, l_id), length_tensor),l_id)
        return self.greedy_decoder._greedy_decode_blank_as_pad_loop_frames(audio_tensor, length_tensor, l_id)
    @torch.inference_mode()
    def forward(self,audio_path,batch_size=2,language="hi"):
        subtitles = []
        for batch, lengths, timestamp in DataLoader(ChunkedData(audio_path),batch_size,shuffle=True,collate_fn=padding_audio):
            batch, lengths = self.encoder(batch, lengths)
            subtitles.extend(make_srt(self.decoding_with_lang(batch,lengths,language),timestamp,self.tokenizer.tokenizers_dict.get(language)))
            torch.cuda.empty_cache()
            gc.collect()
        return srt.compose(subtitles)
