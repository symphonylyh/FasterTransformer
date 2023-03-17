import os,sys
import torch
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../src")
from alexa_teacher_models.models.alexatm_seq2seq import AlexaTMSeq2SeqForConditionalGeneration
from alexa_teacher_models.models.alexatm_seq2seq_config import AlexaTMSeq2SeqConfig


torch.manual_seed(0)

config = AlexaTMSeq2SeqConfig(
    max_position_embeddings=1024,
    d_model=1536, #4096,
    encoder_ffn_dim=1536, #16384
    encoder_layers=2, #46
    encoder_attention_heads=6, #32
    decoder_ffn_dim=1536, #16384
    decoder_layers=2, #32
    decoder_attention_heads=6, #32
)

model = AlexaTMSeq2SeqForConditionalGeneration(config)

encoder_input_ids = torch.tensor([[1, 2, 3]])
decoder_input_ids = torch.tensor([[4, 5]])

fake_output = model(input_ids = encoder_input_ids,
                    decoder_input_ids = decoder_input_ids)


# print("Model = ", model)
print("Output = ", fake_output.logits.data)

for name, para in model.named_parameters():
    print(f'{name}: {para.shape}') 

print(model.model.encoder.token_type_embeddings.weight)