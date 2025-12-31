import gc
import json
import math
from torch import nn
from shruti.rnnt_utils import GreedyRNNTInfer, RNNTDecoder, RNNTJoint
from shruti.torch_weight_conversion import torch_weight_conversion
from shruti.utils import BLANK_ID, ChunkedData, ConvSubsampling, MelPreprocessor, make_srt, padding_audio
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import srt
from torch.utils.data import DataLoader
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerEncoder, Wav2Vec2ConformerConfig


class MaskPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad_mask = None
    def forward(self, hidden_states):
        if self.pad_mask is None:
            return hidden_states
        return hidden_states.float().masked_fill(self.pad_mask.unsqueeze(1), 0.0)


class ShrutiASR(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model = 1024
        ff_expansion_factor = 4
        n_layers = 24
        n_heads = 8
        self.conv_kernel_size = conv_kernel_size = 9
        pred_hidden = 640
        pred_rnn_layers = 2
        joint_hidden = 640
        max_symbols = 10
        conv_channels = 256
        subsampling_factor = 8
        feat_in = 80
        
        self.preprocessor = MelPreprocessor()
        self.pre_encode = ConvSubsampling(subsampling_factor, feat_in, d_model, conv_channels)
        self.encoder = Wav2Vec2ConformerEncoder(
            Wav2Vec2ConformerConfig(
                None, d_model, n_layers, n_heads, d_model*ff_expansion_factor, 
                "swish", conv_depthwise_kernel_size=conv_kernel_size
            )
        )
        
        # Load vocabulary
        self.vocab = json.load(open(hf_hub_download("shethjenil/IndicConformer", "vocab.json")))
        self.language_keys = list(self.vocab.keys())
        
        # Calculate vocab size and create token mappings
        vocab_size = sum([len(v) for v in self.vocab.values()])
        
        # Create token_id_offset mapping (similar to MultilingualTokenizer)
        self.token_id_offset = {}
        offset = 0
        for lang in self.language_keys:
            self.token_id_offset[lang] = offset
            offset += len(self.vocab[lang])
        
        # Create language masks for each token position
        # Each mask indicates which tokens belong to which language
        self.language_masks = {}
        for lang in self.language_keys:
            mask = []
            for token_id in range(vocab_size):
                # Check if token_id falls in this language's range
                lang_start = self.token_id_offset[lang]
                lang_end = lang_start + len(self.vocab[lang])
                mask.append(lang_start <= token_id < lang_end)
            # Add True for blank token at the end
            mask.append(True)
            self.language_masks[lang] = mask
        
        # Decoder
        self.decoder = RNNTDecoder(
            {'pred_hidden': pred_hidden, 'pred_rnn_layers': pred_rnn_layers},
            vocab_size,
            multisoftmax=True,
            language_masks=self.language_masks
        )
        
        # Joint
        self.joint = RNNTJoint(
            {
                'joint_hidden': joint_hidden, 
                'activation': 'relu', 
                'encoder_hidden': d_model, 
                'pred_hidden': pred_hidden
            },
            vocab_size,
            multilingual=True,
            language_keys=self.language_keys,
            language_masks=self.language_masks,
            token_id_offsets=self.token_id_offset,
            preserve_memory=True,
        )
        
        # Greedy Decoder
        self.greedy_decoder = GreedyRNNTInfer(
            self.decoder,
            self.joint,
            BLANK_ID,
            max_symbols
        )
        
        # Load weights
        self.load_state_dict(self.model_preprocess(
            load_file(hf_hub_download("shethjenil/IndicConformer", "model.safetensors"))
        ))
        self.eval()

    def encoder_fn(self, audio_tensor, length_tensor):
        audio_tensor, length_tensor = self.preprocessor.forward(audio_tensor, length_tensor)
        audio_tensor, length_tensor = self.pre_encode.forward(audio_tensor.transpose(1, 2), length_tensor)
        length_tensor = length_tensor.to(torch.int64)
        audio_tensor = audio_tensor * math.sqrt(self.d_model)
        
        # Create padding mask
        self.mask_layer.pad_mask = ~(
            torch.arange(0, audio_tensor.size(1), device=audio_tensor.device)
            .expand(length_tensor.size(0), -1) < length_tensor.unsqueeze(-1)
        )
        
        return self.encoder.forward(audio_tensor).last_hidden_state, length_tensor

    def model_preprocess(self, state_dict):
        change_config = """
preprocessor.featurizer,preprocessor
encoder.pre_encode,pre_encode
norm_feed_forward1,ffn1_layer_norm
norm_feed_forward2,ffn2_layer_norm
feed_forward1.linear1,ffn1.intermediate_dense
feed_forward1.linear2,ffn1.output_dense
feed_forward2.linear1,ffn2.intermediate_dense
feed_forward2.linear2,ffn2.output_dense
norm_self_att,self_attn_layer_norm
norm_out,final_layer_norm
norm_conv,conv_module.layer_norm
fb,mel_fb
.conv.,.conv_module.
pre_encode.conv_module,pre_encode.conv
depthwise_conv,depthwise_conv.1
joint_net.2,joint_net.1
"""
        d_model = self.d_model
        conv_kernel_size = self.conv_kernel_size
        del self.encoder.pos_conv_embed
        self.mask_layer = MaskPad()
        
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].conv_module.depthwise_conv = nn.Sequential(
                self.mask_layer,
                nn.Conv1d(d_model, d_model, conv_kernel_size, 1, 
                         (conv_kernel_size - 1) // 2, 1, d_model)
            )
            self.encoder.layers[i].conv_module.pointwise_conv1 = nn.Conv1d(d_model, 2*d_model, 1)
            self.encoder.layers[i].conv_module.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        
        self.encoder.layer_norm = nn.Identity()
        state_dict: dict[str, torch.Tensor] = torch_weight_conversion(state_dict, change_config)
        state_dict['preprocessor.mel_fb'] = state_dict['preprocessor.mel_fb'].squeeze(0)
        del state_dict['ctc_decoder.decoder_layers.0.bias']
        del state_dict['ctc_decoder.decoder_layers.0.weight']
        return state_dict
    
    @torch.inference_mode()
    def forward(self, audio_path, batch_size=2, language="hi"):
        if language not in self.language_keys:
            raise ValueError(f"Language '{language}' not supported. Available: {self.language_keys}")
        
        subtitles = []
        for batch, lengths, timestamp in DataLoader(ChunkedData(audio_path), batch_size, shuffle=False,collate_fn=padding_audio):
            batch, lengths = self.encoder_fn(batch, lengths)
            hypotheses = self.greedy_decoder._greedy_decode_blank_as_pad_loop_frames(batch, lengths, next(self.parameters()).device,language_ids=[language] * len(lengths))
            subtitles.extend(make_srt(hypotheses, timestamp, self.vocab[language]))
            torch.cuda.empty_cache()
            gc.collect()
        
        return srt.compose(subtitles)