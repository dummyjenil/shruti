import gc
import json
import math
from torch import nn
import torchaudio
from tqdm import tqdm
from shruti.rnnt_utils import GreedyRNNTInfer
from shruti.spleeter import Spleeter
from torch_state_bridge import state_bridge
from shruti.utils import ChunkedData, ConvSubsampling, MaskPad, MelPreprocessor, make_srt, padding_audio
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import srt
from torch.utils.data import DataLoader
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerEncoder, Wav2Vec2ConformerConfig
from pyannote.audio import Pipeline
import os
os.environ['PYANNOTE_SKIP_DEPENDENCY_CHECK'] = '1'

# 'git+https://github.com/dummyjenil/torch-state-bridge.git',
# 'git+https://github.com/dummyjenil/pyannote-audio.git'


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
        blank_id = 256
        conv_channels = 256
        subsampling_factor = 8
        feat_in = 80
        self.preprocessor = MelPreprocessor()
        self.pre_encode = ConvSubsampling(subsampling_factor, feat_in, d_model, conv_channels)
        self.encoder = Wav2Vec2ConformerEncoder(Wav2Vec2ConformerConfig(None, d_model, n_layers, n_heads, d_model*ff_expansion_factor, "swish", conv_depthwise_kernel_size=conv_kernel_size))
        self.vocab = json.load(open(hf_hub_download("shethjenil/IndicConformer", "vocab.json")))
        self.decoder = GreedyRNNTInfer(pred_hidden,pred_rnn_layers,sum([len(v) for v in self.vocab.values()]),d_model,joint_hidden,blank_id,max_symbols,list(self.vocab.keys()))
        self.spleeter = Spleeter(['vocal','instruments'])
        self.speaker_diarization = Pipeline.from_pretrained("shethjenil/speaker-diarization-community-1")
        self.load_state_dict(self.model_preprocess(load_file(hf_hub_download("shethjenil/IndicConformer", "model.safetensors"))))
        self.eval()

    def encoder_fn(self, audio_tensor, length_tensor):
        audio_tensor, length_tensor = self.preprocessor(audio_tensor, length_tensor)
        audio_tensor, length_tensor = self.pre_encode(audio_tensor.transpose(1, 2), length_tensor)
        length_tensor = length_tensor.to(torch.int64)
        audio_tensor = audio_tensor * math.sqrt(self.d_model)
        self.mask_layer.pad_mask = ~(torch.arange(0, audio_tensor.size(1), device=audio_tensor.device).expand(length_tensor.size(0), -1) < length_tensor.unsqueeze(-1))
        return self.encoder(audio_tensor).last_hidden_state, length_tensor

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
decoder.prediction.dec_rnn.lstm,decoder.decoder.lstm
decoder.prediction.embed,decoder.decoder.embed
joint.enc,decoder.joint.enc
joint.pred,decoder.joint.pred
joint.joint_net.2,decoder.joint.joint_net
"""
        d_model = self.d_model
        conv_kernel_size = self.conv_kernel_size
        del self.encoder.pos_conv_embed
        self.mask_layer = MaskPad()
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].conv_module.depthwise_conv = nn.Sequential(self.mask_layer,nn.Conv1d(d_model, d_model, conv_kernel_size, 1, (conv_kernel_size - 1) // 2, 1, d_model))
            self.encoder.layers[i].conv_module.pointwise_conv1 = nn.Conv1d(d_model, 2*d_model, 1)
            self.encoder.layers[i].conv_module.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        
        self.encoder.layer_norm = nn.Identity()
        state_dict: dict[str, torch.Tensor] = state_bridge(state_dict, change_config)
        state_dict['preprocessor.mel_fb'] = state_dict['preprocessor.mel_fb'].squeeze(0)
        del state_dict['ctc_decoder.decoder_layers.0.bias']
        del state_dict['ctc_decoder.decoder_layers.0.weight']
        state_dict.update({"spleeter.stems.vocal."+i:v for i,v in load_file(hf_hub_download("shethjenil/spleeter","2_vocals.safetensors")).items()})
        state_dict.update({"spleeter.stems.instruments."+i:v for i,v in load_file(hf_hub_download("shethjenil/spleeter","2_other.safetensors")).items()})
        return state_dict


    def process(self,input_file):
        audio, sr = torchaudio.load(input_file)
        target_sr = 16000
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        output = self.speaker_diarization({"waveform":audio,"sample_rate":target_sr})
        return {
            "diarization":[[i['start'],i['end'],int(i['speaker'].lstrip("SPEAKER_"))] for i in output.serialize()['diarization']],
            "exclusive_diarization":[[i['start'],i['end'],int(i['speaker'].lstrip("SPEAKER_"))] for i in output.serialize()['exclusive_diarization']],
            "embedding":output.speaker_embeddings.tolist()
        }

    @torch.inference_mode()
    def forward(self, audio_path, batch_size=2, language="hi"):
        subtitles = []
        for batch, lengths, timestamp in tqdm(DataLoader(ChunkedData(audio_path,spleeter=self.spleeter), batch_size, shuffle=False,collate_fn=padding_audio)):
            batch, lengths = self.encoder_fn(batch, lengths)
            subtitles.extend(make_srt(self.decoder(batch, lengths, next(self.parameters()).device,language), timestamp, self.vocab[language]))
            torch.cuda.empty_cache()
            gc.collect()
        return srt.compose(subtitles)
