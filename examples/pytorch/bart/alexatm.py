import os
import sys
ROOT_DIR = os.path.abspath("../../../")
sys.path.append(ROOT_DIR)
lib_path = os.path.join(ROOT_DIR, './build/lib/libth_transformer.so')

import configparser
import numpy as np
import torch
import os
import numpy as np
import time
import math
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration, BartTokenizer 
from transformers import MBartForConditionalGeneration, MBartTokenizer 
from examples.pytorch.bart.utils.ft_encoder import FTBartEncoderWeight, FTBartEncoder
from examples.pytorch.bart.utils.ft_decoding import FTBartDecodingWeight, FTBartDecoding, FTBart

from examples.pytorch.bart.alexa_teacher_models.models.alexatm_seq2seq import AlexaTMSeq2SeqForConditionalGeneration
from examples.pytorch.bart.alexa_teacher_models.models.alexatm_seq2seq_config import AlexaTMSeq2SeqConfig

# specify model name or checkpoint path
model_name = 'alexatm'

# Pytorch
config = AlexaTMSeq2SeqConfig(
    max_position_embeddings=1024,
    d_model=1536, #4096,
    encoder_ffn_dim=1536, #16384
    encoder_layers=2, #46
    encoder_attention_heads=6, #32
    decoder_ffn_dim=1536, #16384
    decoder_layers=2, #32
    decoder_attention_heads=6, #32
    decoder_start_token_id = 2 # unknown? Need Alexa info
)
# print(config)

## random weight model
model = AlexaTMSeq2SeqForConditionalGeneration(config)
model.save_pretrained('/tmp/alexa')

## fixed weight model for debugging
# model = AlexaTMSeq2SeqForConditionalGeneration.from_pretrained('/tmp/alexa')

model = model.eval().to('cuda')

# FT
layernorm_type = "pre_layernorm"
is_mbart = True

# print all weights
# for name, para in model.named_parameters():
#     print(f'{name}: {para.shape}') 

activation_type = config.hidden_act
# note: alexatm LN eps=1e-12
# single-gpu so set TP=1, PP=1
tensor_para_size = 1
pipeline_para_size = 1
bart_with_bias = True
use_gated_activation = False
position_embedding_type = 1 # absolute positional embedding
weight_data_type = np.float32
encoder_head_size = config.d_model // config.encoder_attention_heads
decoder_head_size = config.d_model // config.decoder_attention_heads
remove_padding = False
use_fp16 = True

ft_encoder_weight = FTBartEncoderWeight(
    config,
    tensor_para_size,
    pipeline_para_size,
    bart_with_bias=bart_with_bias,
    mbart=is_mbart,
    use_gated_activation=use_gated_activation,
    position_embedding_type=position_embedding_type,
    weight_data_type=weight_data_type,
)
ft_encoder_weight.load_from_model(model.float())

ft_decoding_weight = FTBartDecodingWeight(
    config,
    tensor_para_size,
    pipeline_para_size,
    bart_with_bias=bart_with_bias,
    mbart=is_mbart,
    use_gated_activation=use_gated_activation,
    position_embedding_type=position_embedding_type,
    weight_data_type=weight_data_type,
)
ft_decoding_weight.load_from_model(model.float())

if use_fp16:
    ft_encoder_weight.to_half()
    ft_decoding_weight.to_half()

ft_encoder = FTBartEncoder(ft_encoder_weight.w, lib_path, config.encoder_attention_heads,
                        encoder_head_size, config.encoder_ffn_dim,
                        config.d_model, remove_padding, config.encoder_layers, 
                        tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size, 
                        bart_with_bias=bart_with_bias, mbart=is_mbart,
                        position_embedding_type=position_embedding_type, 
                        activation_type=activation_type, layernorm_type=layernorm_type)

ft_decoding = FTBartDecoding(ft_decoding_weight.w, lib_path,
                        config.decoder_attention_heads, decoder_head_size,
                        config.decoder_ffn_dim, config.d_model,
                        config.d_model, config.decoder_layers,
                        config.decoder_start_token_id, config.eos_token_id, config.vocab_size,
                        tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size, 
                        bart_with_bias=bart_with_bias, mbart=is_mbart,
                        position_embedding_type=position_embedding_type, 
                        activation_type=activation_type, layernorm_type=layernorm_type)

ft_bart = FTBart(ft_encoder, ft_decoding)

## Test
# batch_size = 1
# input_len = 512
# inputs = {
#     'input_ids': torch.randint(0, config.vocab_size, size=(batch_size, input_len)).to("cuda"),
#     'attention_mask': torch.ones(size=(batch_size, input_len)).to("cuda")    
# }

batch_size = 1
input_len = 12
inputs = {
    'input_ids': torch.tensor([[2, 0, 2005, 340, 10269, 7, 340, 759, 10269, 83,
            12942, 2]]).to("cuda"),
    'attention_mask': torch.ones(size=(batch_size, input_len)).to("cuda")    
}

max_output_len = 32
ft_max_output_len = max_output_len - 2  # to achieve identical results w/ HF, exclude start & end tokens
num_beams = 1
beam_search_diversity_rate = 0.0
topk = None
topp = None
measurement_iters = 10

# PyT test
if use_fp16:
    model.half()
else:
    model.float()
hf_outputs = model.generate(inputs['input_ids'], max_length=max_output_len, num_beams=num_beams)
print("HF output ids",hf_outputs)

hf_latencies = []
for _ in range(measurement_iters):
    start_time = time.time()
    model.generate(inputs['input_ids'], max_length=max_output_len, num_beams=num_beams, use_cache=True)
    end_time = time.time()
    hf_latencies.append(end_time - start_time)
hf_p50 = np.percentile(hf_latencies, 50)
hf_p99 = np.percentile(hf_latencies, 99)
print(f"HF p50: {hf_p50*1000:.2f} ms, p99: {hf_p99*1000:.2f} ms ")

# FT test
return_dict = ft_bart(inputs['input_ids'],
                      inputs['attention_mask'],
                      inputs_embeds=None,
                      beam_size=num_beams,
                      max_seq_len=ft_max_output_len,
                      top_k=topk,
                      top_p=topp,
                      beam_search_diversity_rate=beam_search_diversity_rate,
                      is_return_output_log_probs=False,
                      is_return_cum_log_probs=False)

# ft_bart returns output_ids of shape [batch_size, beam_width, max_output_seq_len]
# ft_bart returns sequence_length of shape [batch_size, beam_width]
ft_output_ids = return_dict['output_ids']
ft_sequence_length = return_dict['sequence_lengths']

ft_outputs = []
for i in range(batch_size):
    # selecting the top sequence from beam width number of sequences
    ft_outputs.append(list(ft_output_ids[i, 0, :][1:ft_sequence_length[i , 0]])) # start from 1 to exclude the 1st token
print("FT output ids", ft_outputs)

ft_latencies = []
for _ in range(measurement_iters):
    start_time = time.time()
    return_dict = ft_bart(inputs['input_ids'],
                          inputs['attention_mask'],
                          inputs_embeds=None,
                          beam_size=num_beams,
                          max_seq_len=ft_max_output_len,
                          top_k=topk,
                          top_p=topp,
                          beam_search_diversity_rate=beam_search_diversity_rate,
                          is_return_output_log_probs=False,
                          is_return_cum_log_probs=False)
    end_time = time.time()
    ft_latencies.append(end_time - start_time)
ft_p50 = np.percentile(ft_latencies, 50)
ft_p99 = np.percentile(ft_latencies, 99)
print(f"FT p50: {ft_p50*1000:.2f} ms, p99: {ft_p99*1000:.2f} ms ")

print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
print(f"Input length: {input_len}, Output length: {max_output_len}")
print(f"HF p50: {hf_p50*1000:.2f} ms, p99: {hf_p99*1000:.2f} ms ")
print(f"FT p50: {ft_p50*1000:.2f} ms, p99: {ft_p99*1000:.2f} ms ")
