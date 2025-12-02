
import math
from pathlib import Path
import re
from typing import List
import torch
from torch import nn
from tqdm import tqdm
from shruti.utils import ConformerLayer, ConvSubsampling, Hypothesis, MelPreprocessor, RelPositionalEncoding, SentencePieceTokenizer, create_masks , RNNTDecoder, RNNTJoint , pack_hypotheses , GreedyBatchedRNNTInfer , MultilingualTokenizer , FilterbankFeatures
def state_loader(state_dict, module_path):
    prefix = module_path + "."
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

blank_id = 256

class ShrutiASR(nn.Module):
    def __init__(self,tokenizer_path):
        super().__init__()
        n_layers = 24
        # TODO bhaai aa real preprocessor nathi FilterbankFeatures aa real chhe teni args joi levi
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
        self.pos_enc.extend_pe(5000)
        pos_bias_u = nn.Parameter(torch.Tensor(8, 1024 // 8))
        pos_bias_v = nn.Parameter(torch.Tensor(8, 1024 // 8))
        nn.init.zeros_(pos_bias_u)
        nn.init.zeros_(pos_bias_v)
        self.layers = nn.ModuleList([ConformerLayer(1024,1024 * 4,8,9,conv_context_size=[4,4],pos_bias_u=pos_bias_u,pos_bias_v=pos_bias_v) for _ in range(n_layers)])
        self.tokenizer = MultilingualTokenizer({i.name:SentencePieceTokenizer(str((i/"tokenizer.model").absolute())) for i in Path(tokenizer_path).rglob("*/")})
        self.decoder = RNNTDecoder({'pred_hidden': 640, 'pred_rnn_layers': 2, 't_max': None, 'dropout': 0.2},self.tokenizer.vocab_size,multisoftmax=True)
        self.joint = RNNTJoint(
            {'joint_hidden': 640, 'activation': 'relu', 'dropout': 0.2, 'encoder_hidden': 1024, 'pred_hidden': 640},
            self.tokenizer.vocab_size,
            vocabulary=self.tokenizer.vocabulary,
            multilingual=True,
            token_id_offsets=self.tokenizer.token_id_offset,
            language_keys=self.tokenizer.langs,
            language_masks={lang: [(token_lang == lang) for _, token_lang in self.tokenizer.langs_by_token_id.items()] + [True] for lang in self.tokenizer.tokenizers_dict},
            offset_token_ids_by_token_id = self.tokenizer.offset_token_ids_by_token_id
            )
        self.decoding = GreedyBatchedRNNTInfer(self.decoder,self.joint,blank_id,10,confidence_method_cfg={'name': 'entropy', 'entropy_type': 'tsallis', 'alpha': 0.33, 'entropy_norm': 'exp', 'temperature': 'DEPRECATED'},)

    def forward(self,audio_tensor,length_tensor,language="hi"):
        with torch.inference_mode():
            # x,sr = torchaudio.load("audio.mp3")
            # if x.shape[0] > 1:
            #     x = x.mean(dim=0)
            # if sr != 16000:
            #     x = torchaudio.functional.resample(x, sr, 16000)
            # x = x.unsqueeze(0) # ADD BATCH DIM
            # l = torch.tensor(x.shape[-1]).unsqueeze(0) # ADD BATCH DIM
            
            audio_tensor,length_tensor = self.preprocessor.forward(audio_tensor,length_tensor)
            audio_tensor,length_tensor = self.pre_encode.forward(audio_tensor.transpose(1, 2),length_tensor)
            audio_tensor , pos_emb = self.pos_enc.forward(audio_tensor)
            length_tensor = length_tensor.to(torch.int64)
            pad_mask, att_mask = create_masks(length_tensor,audio_tensor.size(1),audio_tensor.device)
            for i in tqdm(self.layers):
                audio_tensor = i.forward(audio_tensor,att_mask,pos_emb,pad_mask)
            l_id = [language] * len(audio_tensor)
            return self.decode_hypothesis(pack_hypotheses(self.decoding._greedy_decode_blank_as_pad_loop_frames(audio_tensor, length_tensor, audio_tensor.device, language_ids=l_id), length_tensor),l_id)
    def decode_hypothesis(self,hypotheses_list: List[Hypothesis], lang_ids: List[str] = None) -> List[Hypothesis]:
        for ind in range(len(hypotheses_list)):
            prediction = hypotheses_list[ind].y_sequence
            if type(prediction) != list:
                prediction = prediction.tolist()
            prediction = [p for p in prediction if p != blank_id]
            if lang_ids is not None: #CTEMO
                hypothesis = self.tokenizer.ids_to_text(prediction, lang_ids[ind])
            else:
                hypothesis = re.sub(r'(\s+)([\.\,\?])', r'\2', self.tokenizer.ids_to_text(prediction))
            hypotheses_list[ind].text = hypothesis
        return hypotheses_list
    def load_model(self,model_path):
        s = torch.load(model_path)
        # self.preprocessor.load_state_dict(state_loader(s,"model.preprocessor"))
        self.pre_encode.load_state_dict(state_loader(s,"model.encoder.pre_encode"))
        self.layers.load_state_dict(state_loader(s,"model.encoder.layers"))
        self.decoder.load_state_dict(state_loader(s,"model.decoder"))
        self.joint.load_state_dict(state_loader(s,"model.joint"))
