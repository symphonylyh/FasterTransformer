# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import torch  # pytype: disable=import-error
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../../")
from examples.pytorch.bart.alexa_teacher_models.models.alexatm_seq2seq import AlexaTMSeq2SeqForConditionalGeneration
from examples.pytorch.bart.alexa_teacher_models.models.alexatm_seq2seq_config import AlexaTMSeq2SeqConfig

import argparse
import configparser
from datetime import datetime
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)

rename_mapping = {"relative_attention_num_buckets": "relative_attention_num_buckets_or_max_pos_seq_len"}

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"

def np_to_pyt_data_type(data_type):
    if data_type == np.float32:
        return torch.float32
    elif data_type == np.float16:
        return torch.float16
    else:
        assert False, f"Invalid weight data type {data_type}"

def fuse_decoder_qkv(model, factor, saved_dir, np_weight_data_type):
    model_dict = {}
    for name, param in model.named_parameters():
        name = name.replace("model.", "")
        if name.find("decoder") != -1 and name.find("self_attn") != -1:
            model_dict[name] = param

    for i in range(model.model.decoder.config.decoder_layers):
        shape = model_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"].T.shape
        qkv = torch.cat([model_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"].T,
                         model_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"].T,
                         model_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"].T], dim=-1)
        
        shape_bias = model_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"].shape
        qkv_bias = torch.cat([model_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"],
                         model_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"],
                         model_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"]], dim=-1)

        qkv = qkv.reshape([shape[0], 3, shape[1]])
        qkv = qkv.cpu().detach().numpy().astype(np_weight_data_type)

        qkv_bias = qkv_bias.reshape([3, shape_bias[0]])
        qkv_bias = qkv_bias.cpu().detach().numpy().astype(np_weight_data_type)

        split_qkvs = np.split(qkv, factor, axis=-1)
        split_qkv_biases = np.split(qkv_bias, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"decoder.layers.{i}.self_attn.qkv.weight.{j}.bin"
            split_qkvs[j].tofile(saved_path.as_posix())
            saved_path = saved_dir / f"decoder.layers.{i}.self_attn.qkv.bias.{j}.bin"
            split_qkv_biases[j].tofile(saved_path.as_posix())


def split_and_convert_process(model, factor, saved_dir, np_weight_data_type):
    for name, param in model.state_dict().items():
        # HF BART/mBART model's weight names are prepended with "model.", remove for consistency
        name = name.replace("model.", "")
        
        # transpose all 2D weights, EXCEPT the embedding tables (handled next)
        if param.dim() == 2:  
            param = param.transpose(1, 0)

        param = param.cpu().detach().numpy().astype(np_weight_data_type)
        saved_name = name
        LOGGER.debug(f"name: {name}, param.shape: {param.shape}")

        # word embedding table [vocab size, hidden size] (1) should NOT be transposed (2) may be scaled (mBART), instead of customize this in FT, it's better to modify during embedding loading
        if name.find("shared.weight") != -1:
            param = param.transpose(1, 0)
            embedding_scale = 1.0 #np.sqrt(model.config.d_model) if model.config.scale_embedding else 1.0
            param = param * embedding_scale
            saved_path = saved_dir / f"{saved_name}.bin"
            param.tofile(saved_path.as_posix())

        # positional embedding table [max position embeddings, hidden size] (1) should NOT be transposed (2) need to apply offset of 2 for absolute position embeddings in BART/mBART for decoder. Encoder is not using BART/mBART so NO need to offset 2
        elif name.find("encoder.position_embeddings") != -1: 
            param = param.transpose(1, 0)
            saved_path = saved_dir / f"{saved_name}.bin"
            param.tofile(saved_path.as_posix())
        elif name.find("decoder.embed_positions") != -1:
            param = param.transpose(1, 0)[2:, :]
            saved_path = saved_dir / f"{saved_name}.bin"
            param.tofile(saved_path.as_posix())
        
        # token type embedding table should NOT be transposed, [type vocab size, hidden size]
        elif name.find("encoder.token_type_embeddings") != -1:
            param = param.transpose(1, 0)
            saved_path = saved_dir / f"{saved_name}.bin"
            param.tofile(saved_path.as_posix())
        
        # output embedding table [vocab, hidden size] (1) should NOT be transposed
        elif name.find("lm_head.weight") != -1:
            param = param.transpose(1, 0)
            saved_path = saved_dir / f"{saved_name}.bin"
            param.tofile(saved_path.as_posix())

        # embedding bias aka final_logits_bias (may not exist, keys to ignore)
        elif name.find("final_logits_bias") != -1:
            saved_path = saved_dir / f"{saved_name}.bin"
            param.tofile(saved_path.as_posix())

        # all layernorm's weight and bias are shared weights, only need to convert the weights of rank 0
        # type 1 - layer-level LN: {layers}_layer_norm{.weight, .bias} 
        # type 2 - transformer-level LN after transformer (special in mBART): {encoder}layer_norm{.weight, .bias}, {decoder}layernorm_output{.weight, .bias} 
        elif name.find("layer_norm") != -1 or name.find("layernorm_output") != -1:
            saved_path = saved_dir / f"{saved_name}.bin"
            param.tofile(saved_path.as_posix())

        # FC1 layers weights and biases & encoder self-attn Q/K/V weights and biases & decoder cross-attn Q/K/V weights and biases, split on last dim
        elif (
            name.find("fc1") != -1 or 
            name.find("intermediate.dense") != -1
            or (name.find("encoder") != -1 and (
                name.find("self.query") != -1
                or name.find("self.key") != -1
                or name.find("self.value") != -1
            )
            )
            or name.find("encoder_attn.q_proj") != -1
            or name.find("encoder_attn.k_proj") != -1
            or name.find("encoder_attn.v_proj") != -1
        ):
            split_params = np.split(param, factor, axis=-1)
            for j in range(factor):
                saved_path = saved_dir / f"{saved_name}.{j:d}.bin"
                split_params[j].tofile(saved_path.as_posix())

        # skip decoder self-attn Q/K/V weights and biases which will be fused later
        elif (
            name.find("decoder") != -1 and
            (
                name.find("self_attn.q_proj") != -1
                or name.find("self_attn.k_proj") != -1
                or name.find("self_attn.v_proj") != -1
            )
        ):
            pass
        
        # output linear layers weights & FC2 layers weights, split on first dim
        elif (
            name.find("attention.output.dense.weight") != -1
            or name.find("encoder_attn.out_proj.weight") != -1
            or name.find("self_attn.out_proj.weight") != -1
            or name.find("output.dense.weight") != -1
            or name.find("fc2.weight") != -1
        ):
            split_params = np.split(param, factor, axis=0)
            for j in range(factor):
                saved_path = saved_dir / f"{saved_name}.{j:d}.bin"
                split_params[j].tofile(saved_path.as_posix())

        # output linear layers biases & FC2 layers biases are shared, no split
        elif (
            name.find("attention.output.dense.bias") != -1
            or name.find("out_proj.bias") != -1
            or name.find("output.dense.bias") != -1
            or name.find("fc2.bias") != -1
        ):
            saved_path = saved_dir / f"{saved_name}.bin"
            param.tofile(saved_path.as_posix())

        elif name.find("encoder.token_embeddings.weight") != -1 or \
                name.find("decoder.embed_tokens.weight") != -1:
            LOGGER.warning(f"Not save {name}, using shared.weight directly.")
        elif name.find("encoder.position_ids") != -1:
            LOGGER.warning(f"Not save {name}, position ids is deterministic.")
        else:
            LOGGER.warning(f"cannot find name '{name}' with shape {param.shape}")

def convert_checkpoint(args):
    saved_dir = Path(args.saved_dir) / f"{args.inference_tensor_para_size:d}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    if 'alexatm' in args.in_file:
        config = AlexaTMSeq2SeqConfig(
            max_position_embeddings=1024,
            d_model=1536, #4096,
            encoder_ffn_dim=1536, #16384
            encoder_layers=2, #46
            encoder_attention_heads=6, #32
            decoder_ffn_dim=1536, #16384
            decoder_layers=2, #32
            decoder_attention_heads=6, #32,
            decoder_start_token_id=2
        )
        bart_model = AlexaTMSeq2SeqForConditionalGeneration(config)

    # print weight names
    # for name, para in bart_model.state_dict().items():
    #     print(f'{name}: {para.shape}') 
    
    # sample input verification with Triton
    bart_model.to('cuda')
    batch_size = 1
    input_len = 12
    inputs = {
        'input_ids': torch.tensor([[2, 0, 2005, 340, 10269, 7, 340, 759, 10269, 83,
                12942, 2]]).to("cuda"),
        'attention_mask': torch.ones(size=(batch_size, input_len)).to("cuda")    
    }
    max_output_len = 32
    num_beams = 1
    hf_outputs = bart_model.generate(inputs['input_ids'], max_length=max_output_len, num_beams=num_beams)
    print("HF: ", hf_outputs)

    config = configparser.ConfigParser()
    
    config["encoder"] = {}
    config["decoder"] = {}
    config["structure"] = {}

    # only save useful hyper-params for FT
    for key, val in bart_model.model.encoder.config.to_dict().items():
        if key in {'encoder_ffn_dim', 'encoder_layers', 'encoder_attention_heads', 'relative_attention_num_buckets'}: 
            config["encoder"][key] = f"{val}"

    for key, val in bart_model.model.decoder.config.to_dict().items():
        if key in {'decoder_ffn_dim', 'decoder_layers', 'decoder_attention_heads', 'relative_attention_num_buckets', 'tie_word_embeddings', 'bos_token_id', 'pad_token_id', 'eos_token_id', 'sep_token_id', 'decoder_start_token_id'}: 
            config["decoder"][key] = f"{val}"
    
    config["decoder"]["decoder_start_token_id"] = "2" # unknown? Need Alexa info

    for key, val in rename_mapping.items():
        if key in config['encoder']:
            config['encoder'][val] = config['encoder'].pop(key)
        if key in config['decoder']:
            config['decoder'][val] = config['decoder'].pop(key)

    # common params in encoder & decoder
    for part in {'encoder', 'decoder'}:
        config[part]["vocab_size"] = f"{bart_model.config.vocab_size}"
        config[part]["d_model"] = f"{bart_model.config.d_model}"
        config[part]["_name_or_path"] = f"{bart_model.config._name_or_path}"
        config[part]["architectures"] = f"{bart_model.config.architectures}"
        config[part]["transformers_version"] = f"{bart_model.config.transformers_version}"
        config[part]["model_type"] = f"{bart_model.config.model_type}"
        config[part]["weight_data_type"] = args.weight_data_type

    # structure info
    config["structure"]["bart_with_bias"] = "true"
    config["structure"]["mbart"] = "true"
    config["structure"]["activation_function"] = f"{bart_model.config.hidden_act}"
    if config["structure"]["activation_function"].find("gated") != -1:
        config["structure"]["use_gated_activation"] = "true"
    config["structure"]["position_embedding_type"] = "absolute"

    with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)
   
    np_weight_data_type = get_weight_data_type(args.weight_data_type)
    i_gpu_num = args.inference_tensor_para_size

    split_and_convert_process(bart_model, i_gpu_num, saved_dir, np_weight_data_type)

    # fuse QKV weights and biases for decoder self attention
    fuse_decoder_qkv(bart_model, i_gpu_num, saved_dir, np_weight_data_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-saved_dir", "-o", type=str, help="file name of output file", required=True)
    parser.add_argument("-in_file", "-i", type=str, help="file name of input checkpoint file or name of HuggingFace model", required=True)
    parser.add_argument("-inference_tensor_para_size", "-i_g", type=int,
                        help="How many gpus for inference", required=True)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--verbose", action="store_true", help="Provide verbose messages")
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=log_format)
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))