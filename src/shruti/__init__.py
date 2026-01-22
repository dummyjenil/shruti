import gc
import math
from torch import nn
import torchaudio
from tqdm import tqdm
from torch_state_bridge import state_bridge
from shruti.utils import ChunkedData, ConvSubsampling, GreedyRNNTInfer, MaskPad, MelPreprocessor, RNNTDecoder, RNNTJoint, make_srt, padding_audio , calc_length
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import srt
from torch.utils.data import DataLoader
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerEncoder, Wav2Vec2ConformerConfig


class ShrutiASR(nn.Module):
    def __init__(self,language=None):
        super().__init__()
        langs = ['as', 'bn', 'brx', 'doi', 'gu', 'hi', 'kn', 'kok', 'ks', 'mai', 'ml', 'mni', 'mr', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur']
        assert language in langs + [None]
        self.d_model = d_model = 512 if language else 1024
        n_layers = 17 if language else 24
        pred_rnn_layers = 1 if language else 2
        conv_channels = 512 if language else 256
        self.conv_kernel_size = conv_kernel_size = 31 if language else 9
        subsampling_factor = 4 if language else 8
        pred_hidden = 640
        self.joint_hidden = joint_hidden = 640
        blank_id = 256
        vocab_size = 5632
        lang_vocab_size = 256
        self.preprocessor = MelPreprocessor()
        self.pre_encode = ConvSubsampling(int(math.log(subsampling_factor, 2)), d_model, conv_channels,not(language))
        self.encoder = Wav2Vec2ConformerEncoder(Wav2Vec2ConformerConfig(None, d_model, n_layers, 8, d_model*4, "swish", conv_depthwise_kernel_size=conv_kernel_size))

        if not language:
            self.vocab = {lang:["<unk>"]+[i.lstrip("##") if i.startswith("##") else "▁"+i for i in open(hf_hub_download("shethjenil/IndicConformer", f"all/tokenizer/{lang}/vocab.txt")).read().split("\n")] for lang in langs}
            self.lang_joint_net = nn.ModuleDict({i:nn.Linear(joint_hidden,lang_vocab_size+1) for i in langs})
        else:
            self.vocab = {language:["<unk>"]+[i.lstrip("##") if i.startswith("##") else "▁"+i for i in open(hf_hub_download("shethjenil/IndicConformer", f"{language}/tokenizer/{language}/vocab.txt")).read().split("\n")]}

        self.decoder = GreedyRNNTInfer(RNNTDecoder(vocab_size,pred_hidden,pred_rnn_layers),RNNTJoint(d_model,pred_hidden,joint_hidden,lang_vocab_size+1),blank_id)
        self.scaler = math.sqrt(self.d_model)
        self.denormalizer = subsampling_factor * self.preprocessor.hop_length / self.preprocessor.sr
        self.language = language
        self.load_state_dict(self.model_preprocess(load_file(hf_hub_download("shethjenil/IndicConformer", f"{language if language else "all"}/model.safetensors"))))
        self.eval()

    def asr_fn(self, audio_tensor, length_tensor):
        audio_tensor, length_tensor = self.preprocessor(audio_tensor, length_tensor)
        audio_tensor = self.pre_encode(audio_tensor) * self.scaler
        length_tensor = calc_length(length_tensor,2,3,2,self.pre_encode._sampling_num).to(torch.int64)
        self.mask_layer.pad_mask = ~(torch.arange(audio_tensor.size(1), device=audio_tensor.device).expand(length_tensor.size(0), -1) < length_tensor.unsqueeze(-1))
        return self.decoder(self.encoder(audio_tensor).last_hidden_state, length_tensor)

    def model_preprocess(self, state_dict):
        d_model = self.d_model
        conv_kernel_size = self.conv_kernel_size
        self.mask_layer = MaskPad()
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].conv_module.depthwise_conv = nn.Sequential(self.mask_layer,nn.Conv1d(d_model, d_model, conv_kernel_size, 1, (conv_kernel_size - 1) // 2, 1, d_model))
            self.encoder.layers[i].conv_module.pointwise_conv1 = nn.Conv1d(d_model, 2*d_model, 1)
            self.encoder.layers[i].conv_module.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.encoder.layer_norm = nn.Identity()
        del self.encoder.pos_conv_embed
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
joint.enc,decoder.joint.enc_proj
joint.pred,decoder.joint.pred_proj
joint.joint_net.2,lang_joint_net
"""
        state_dict: dict[str, torch.Tensor] = state_bridge(state_dict, change_config)
        if not self.language:
            state_dict["decoder.joint.joint_net.weight"] = self.decoder.joint.joint_net.weight
            state_dict["decoder.joint.joint_net.bias"] = self.decoder.joint.joint_net.bias
        else:
            state_dict["decoder.joint.joint_net.weight"] = state_dict.get(f'lang_joint_net.{self.language}.weight').clone()
            state_dict["decoder.joint.joint_net.bias"] = state_dict.get(f'lang_joint_net.{self.language}.bias').clone()
            state_dict = {k: v for k, v in state_dict.items() if "lang_joint_net" not in k}
        state_dict['preprocessor.mel_fb'] = state_dict['preprocessor.mel_fb'].squeeze(0)

        del state_dict['ctc_decoder.decoder_layers.0.bias']
        del state_dict['ctc_decoder.decoder_layers.0.weight']
        
        return state_dict

    @torch.inference_mode()
    def forward(self, audio_path, batch_size=2, language=None,use_tqdm=False):
        if not self.language:
            self.decoder.joint.joint_net.load_state_dict(self.lang_joint_net[language].state_dict())

        device = next(self.parameters()).device
        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, sr, self.preprocessor.sr)
        subtitles = []
        loader = DataLoader(ChunkedData(wav,self.preprocessor.sr), batch_size, shuffle=False,collate_fn=padding_audio)
        if use_tqdm:
            pbar = tqdm(total=len(self.encoder.layers)*len(loader))
            def hook_fn(module, inp, out):pbar.update(1)
            hooks = [layer.register_forward_hook(hook_fn) for layer in self.encoder.layers]

        for batch, lengths, timestamp in loader:
            hyp = self.asr_fn(batch.to(device), lengths.to(device))
            subtitles.extend(make_srt(hyp, timestamp.to(device), self.vocab[self.language if self.language else language],self.denormalizer))
            torch.cuda.empty_cache()
            gc.collect()
            yield srt.compose(subtitles)
        if use_tqdm:
            for h in hooks:h.remove()
            pbar.close()
