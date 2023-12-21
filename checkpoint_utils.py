

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_all(encoders, decoders, prefix, dec_prefix=None):
    
    for l, enc in encoders.items():
       torch.save(enc.state_dict(), prefix + "-enc-" + l + ".pt")

    for l, dec in decoders.items():
       torch.save(dec.state_dict(), prefix + "-dec-" + l + ".pt")



def load_models(enc_model, enc_params, enc_langs, dec_model, dec_params, dec_langs, prefix, dec_prefix=None, device=device):
    encoders = {}
    decoders = {}

    dec_prefix = prefix if dec_prefix is None else dec_prefix

    
    for el in enc_langs:
        model = enc_model(**enc_params[el])
        model.load_state_dict(torch.load(prefix + "-enc-" + el + ".pt", map_location=device))
        encoders[el] = model.to(device)


    for dl in dec_langs:
        model = dec_model(**dec_params[dl])
        model.load_state_dict(torch.load(dec_prefix + "-dec-" + dl + ".pt", map_location=device))
        decoders[dl] = model.to(device)


    return encoders, decoders



