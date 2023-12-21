

from tqdm import tqdm
import math
import torch
import torch.nn.functional as F

from checkpoint_utils import save_all, load_models
from evaluation import log_loss


device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

class TrainConfig:
    def __init__(self, data, enc_model, enc_params, dec_model, dec_params, 
                 opt_type, loss_fn, enc_opt_params={}, dec_opt_params={}, 
                 num_epochs=1, eval_inter=math.inf, decode_tokenizers=None, 
                 num_batches=-1, load_model=False, load_prefix=None,
                 evaluator=None, save_models=False, save_prefix="", 
                 save_inter=1e8, log_loss=False, log_file=""):

        self.data = data

        if not load_model:
            self.encoders = {l: enc_model(**enc_params[l]).to(device) for l in data.langs}
            self.decoders = {l: dec_model(**dec_params[l]).to(device) for l in data.langs}
        else:
            self.encoders, self.decoders = load_models(enc_model, enc_params, data.langs, dec_model, dec_params, data.langs, load_prefix, device=device)



        self.enc_opts = {l: opt_type(self.encoders[l].parameters(), **enc_opt_params) for l in data.langs}
        self.dec_opts = {l: opt_type(self.decoders[l].parameters(), **dec_opt_params) for l in data.langs}

        self.loss_fn = loss_fn

        self.num_epochs = num_epochs
        self.eval_inter = eval_inter
        self.decode_tokenizers = decode_tokenizers
        self.num_batches = data.num_batches
        self.evaluator = evaluator
        self.save_models = save_models
        self.save_prefix = save_prefix
        self.save_inter = save_inter
        self.log_loss = log_loss
        self.log_file = log_file


class SimpleTrainConfig:
    def __init__(self, data, loss_fn, num_epochs=1, eval_inter=math.inf, decode_tokenizers=None, 
                 num_batches=-1, evaluator=None, save_models=False, save_prefix="", 
                 save_inter=1e8, log_loss=False, log_file=""):
        
        self.data = data
        self.loss_fn = loss_fn

        self.num_epochs = num_epochs
        self.eval_inter = eval_inter
        self.decode_tokenizers = decode_tokenizers
        self.num_batches = data.num_batches
        self.evaluator = evaluator
        self.save_models = save_models
        self.save_prefix = save_prefix
        self.save_inter = save_inter
        self.log_loss = log_loss
        self.log_file = log_file





def train(train_config):
    # data: DataLoader
    # encoders/decoders: dict of encoder and decoder models with language keys
    # enc_opt/dec_opt:   dict of optimizers for the encoders and decoders (languages keys)
    # train_config:   for additional training paramters (can eventually hold stuff like eval intervals etc.)


    data = train_config.data
    encoders = train_config.encoders
    decoders = train_config.decoders
    enc_opts = train_config.enc_opts
    dec_opts = train_config.dec_opts
    loss_fn = train_config.loss_fn

    if train_config.evaluator != None:
        train_config.evaluator.evaluate(encoders, decoders)



    for epoch in range(train_config.num_epochs):
        step = 0
        cum_loss = 0

        for src_lang, tgt_lang, src, tgt in tqdm(data, total=train_config.num_batches):


            src = src.to(device)
            tgt = tgt.to(device)

            enc = encoders[src_lang]
            dec = decoders[tgt_lang]

            enc_opt = enc_opts[src_lang]
            dec_opt = dec_opts[tgt_lang]

            enc_opt.zero_grad()
            dec_opt.zero_grad()

            enc_out = enc(src)
            dec_out = dec(tgt[:, :-1], enc_out)

            batch, seq, dim = dec_out.shape

            pred = torch.reshape(dec_out, (batch*seq, dim))
            targets = torch.reshape(tgt[:, 1:], (batch*seq,))

            loss = loss_fn(pred, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(enc.parameters(), 2)
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 2)

            enc_opt.step()
            dec_opt.step()

            cum_loss += loss.item()

            if step % train_config.eval_inter == train_config.eval_inter - 1:
                with torch.no_grad():
                    probs = F.softmax(dec_out[0], dim=-1)
                    idxs = torch.multinomial(probs, num_samples=1)

                    gen_text = train_config.decode_tokenizers[tgt_lang].batch_decode(idxs)
                    tgt_text = train_config.decode_tokenizers[tgt_lang].batch_decode(tgt[0])
                    src_text = train_config.decode_tokenizers[src_lang].batch_decode(src[0])

                    print(src_lang, tgt_lang)
                    print(cum_loss / train_config.eval_inter)
                    log_loss(cum_loss / train_config.eval_inter, train_config.log_file)
                    cum_loss = 0
                    print("\n\n" + " ".join(gen_text[0:50]))
                    print("\n" + " ".join(tgt_text[0:50]))
                    print("\n" + " ".join(src_text[0:50]))
                    print("\n\n")
                    
                    if train_config.evaluator != None:
                        train_config.evaluator.evaluate(encoders, decoders)
                    
                    print("#######################################")


            if train_config.save_models and step % train_config.save_inter == train_config.save_inter - 1:
                save_all(encoders, decoders, train_config.save_prefix)


            step += 1

        print("EPOCH: ", epoch + 1)






def new_encoder(train_config, enc, enc_opt, decoders, lang):
    # data: DataLoader
    # encoders/decoders: dict of encoder and decoder models with language keys
    # enc_opt/dec_opt:   dict of optimizers for the encoders and decoders (languages keys)
    # train_config:   for additional training paramters (can eventually hold stuff like eval intervals etc.)


    data = train_config.data
    loss_fn = train_config.loss_fn

    if train_config.evaluator != None:
        train_config.evaluator.evaluate({lang: enc}, decoders)



    for epoch in range(train_config.num_epochs):
        step = 0
        cum_loss = 0

        for src_lang, tgt_lang, src, tgt in tqdm(data, total=train_config.num_batches):

            if src_lang != lang:
                raise KeyError("src lang does not match encoder")

            src = src.to(device)
            tgt = tgt.to(device)

            dec = decoders[tgt_lang]            

            enc_opt.zero_grad()

            enc_out = enc(src)
            dec_out = dec(tgt[:, :-1], enc_out)

            batch, seq, dim = dec_out.shape

            pred = torch.reshape(dec_out, (batch*seq, dim))
            targets = torch.reshape(tgt[:, 1:], (batch*seq,))

            loss = loss_fn(pred, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(enc.parameters(), 2)

            enc_opt.step()

            cum_loss += loss.item()

            if step % train_config.eval_inter == train_config.eval_inter - 1:
                with torch.no_grad():
                    probs = F.softmax(dec_out[0], dim=-1)
                    idxs = torch.multinomial(probs, num_samples=1)

                    gen_text = train_config.decode_tokenizers[tgt_lang].batch_decode(idxs)
                    tgt_text = train_config.decode_tokenizers[tgt_lang].batch_decode(tgt[0])
                    src_text = train_config.decode_tokenizers[src_lang].batch_decode(src[0])

                    print(src_lang, tgt_lang)
                    print(cum_loss / train_config.eval_inter)
                    log_loss(cum_loss / train_config.eval_inter, train_config.log_file)
                    cum_loss = 0
                    print("\n\n" + " ".join(gen_text[0:50]))
                    print("\n" + " ".join(tgt_text[0:50]))
                    print("\n" + " ".join(src_text[0:50]))
                    print("\n\n")
                    
                    if train_config.evaluator != None:
                        train_config.evaluator.evaluate({lang: enc}, decoders)
                    
                    print("#######################################")


            if train_config.save_models and step % train_config.save_inter == train_config.save_inter - 1:
                save_all({lang: enc}, {}, train_config.save_prefix)


            step += 1

        print("EPOCH: ", epoch + 1)







