
import os
os.environ['TRANSFORMERS_CACHE'] = '/LLM/comp550/model_cache'

import copy

from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn

device = "cuda"

model_name = "t5-small"

print("Loading")
#base_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
config = AutoConfig.from_pretrained(model_name)
base_model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(model_name)


#print(list(base_model.named_modules())[:3])


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = copy.deepcopy(base_model.encoder)

    def forward(self, x):
        out = self.layers.forward(x).last_hidden_state
        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.layers = copy.deepcopy(base_model.decoder)

        self.ff = nn.Linear(512, vocab_size)


    def forward(self, input_ids, encoder_out):
        out = self.layers.forward(input_ids=input_ids, encoder_hidden_states=encoder_out)
        
        return self.ff(out.last_hidden_state.float())


if __name__ == "__main__":

    enc_model = Encoder()
    dec_model = Decoder()

    test = "This is a test"

    input_ids = tokenizer(test, return_tensors='pt').input_ids

    enc_out = enc_model(input_ids)

    dec_out = dec_model(tokenizer("", return_tensors='pt').input_ids, enc_out.last_hidden_state)

    print(dec_out.last_hidden_state.shape)

    id = torch.argmax(dec_out.last_hidden_state.squeeze().squeeze())

    print(id)

    print(tokenizer.decode(id.item())[0])


    base_out = torch.argmax(base_model(input_ids, decoder_input_ids=tokenizer("", return_tensors='pt').input_ids).last_hidden_state.squeeze().squeeze())

    print(tokenizer.decode(base_out.item())[0])



