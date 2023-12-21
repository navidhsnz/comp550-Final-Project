

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"



class ToyEvaluator:
    def __init__(self, tokenizers, test_phrase):

        self.tokenizers = tokenizers
        self.test_phrase_ids = tokenizers["en"](test_phrase, return_tensors="pt").input_ids

    def generate(self, decoder, enc_out, bos, eos, max_len=256):

        cur_seq = bos
        for i in range(max_len):
            
            dec_in = torch.tensor(cur_seq).to(device).unsqueeze(0)
            dec_out = decoder(dec_in, enc_out).squeeze()
            
            if len(cur_seq) > 1:
                next_tok = torch.argmax(dec_out[-1])
            else:
                next_tok = torch.argmax(dec_out) 
            
            cur_seq.append(next_tok.item())

            if cur_seq[:-1] == eos:
                return cur_seq

        return cur_seq

    
    def evaluate(self, encoders, decoders):

        enc_out = encoders["en"](self.test_phrase_ids.to(device))

        for l, decoder in decoders.items():
            #bos = [self.tokenizers[l].bos_token_id, self.tokenizers[l](" ").input_ids[1]]
            bos = [self.tokenizers[l]("").input_ids[0]]

            generated = self.generate(decoder, enc_out, bos, self.tokenizers[l].eos_token_id)

            print("Decoding: ", l)
            print(self.tokenizers[l].decode(generated[:20]))


class EvalModel:
    def __init__(self, encoder, decoder, enc_tokenizer, dec_tokenizer):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer


    def generate(self, src, init, max_len):
        
        enc_out = self.encoder(src)


        cur_seq = init
        for i in range(max_len):
            dec_in = torch.tensor(cur_seq).to(device).unsqueeze(0)
            dec_out = self.decoder(dec_in, enc_out).squeeze()
            
            if len(cur_seq) > 1:
                next_tok = torch.argmax(dec_out[-1])
            else:
                next_tok = torch.argmax(dec_out) 
            
            cur_seq.append(next_tok.item())

            if cur_seq[-1] == self.dec_tokenizer.eos_token_id:
                return cur_seq

        return cur_seq

    
    def translate(self, src):

        src_tok = self.enc_tokenizer(src, return_tensors="pt").input_ids.to(device)
        init = [self.dec_tokenizer.bos_token_id]

        gen_out = self.generate(src_tok, init, 256)
        gen_text = self.dec_tokenizer.decode(gen_out)

        return gen_text




            

def log_loss(loss, log_file):
    with open(log_file, 'a') as f:
        f.write(str(loss)+"\n")
            







