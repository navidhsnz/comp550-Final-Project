
import os
os.environ["HF_DATASETS_CACHE"] = "./hf_datasets"
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"

import torch.optim as opt
import torch.nn as nn

from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

from un_pc_loader import load_un_pc
from data_manage import DataLoader
from evaluation import ToyEvaluator, EvalModel
from training import *
from checkpoint_utils import load_models
#from encoder_decoder import Encoder, Decoder
from basic_transformer import Encoder, Decoder
#from lstm_enc_dec import Encoder, Decoder

from eval_metrics import *




def count_parameters(model_type, args):
    model = model_type(**args)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



en_tok = AutoTokenizer.from_pretrained("t5-small")
fr_tok = AutoTokenizer.from_pretrained("qanastek/pos-french-camembert")
es_tok = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
ru_tok = AutoTokenizer.from_pretrained("blinoff/roberta-base-russian-v0")
ar_tok = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
zh_tok = AutoTokenizer.from_pretrained("bert-base-chinese")

tokenizers = {"en": en_tok, "fr": fr_tok, "es": es_tok, "ru": ru_tok, "ar": ar_tok, "zh": zh_tok}



for l, t in tokenizers.items():
    t.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>"}, replace_additional_special_tokens=True)
    t._tokenizer.post_processor = TemplateProcessing(
    single=t.bos_token + " $A " + t.eos_token,
    special_tokens=[(t.eos_token, t.eos_token_id), (t.bos_token, t.bos_token_id)],
)



def train_diff_families():
    data = load_un_pc([("ar", "en"), ("ar", "ru"), ("en", "ru")], tokenizers, 2300000, self_map=False)

    data_loader = DataLoader(data, 16)

    print(sum(data_loader.nums_examples.values()) )
    print(data_loader.nums_examples)


    enc_model = Encoder
    dec_model = Decoder

    dec_params = {l: {"vocab_size": len(tokenizers[l]), "pad_token": tokenizers[l].pad_token_id} for l in data_loader.langs}

    enc_params = {l: {"vocab_size": len(tokenizers[l]), "pad_token": tokenizers[l].pad_token_id} for l in data_loader.langs}


    evaluator = ToyEvaluator(tokenizers, "These are human rights")

    train_conf = TrainConfig(data_loader, enc_model, enc_params, dec_model, dec_params, opt.Adam, nn.CrossEntropyLoss(), enc_opt_params={"lr": 0.0001}, dec_opt_params={"lr": 0.0001}, eval_inter=1000, decode_tokenizers=tokenizers, num_epochs=2000, evaluator=evaluator, save_models=True, save_prefix="./checkpoints/diff_families/second", save_inter=10000, log_loss=True, log_file = "./loss_log.txt")


    print("PARAMETERS: ", count_parameters(dec_model, dec_params["en"]))

    train(train_conf)

    # 160k batches



def new_enc_one_pair():
    data = load_un_pc([("ar", "zh")], tokenizers, 1700000, self_map=False, do_reverse=False, swap_pairs=True)

    data_loader = DataLoader(data, 16)

    print(sum(data_loader.nums_examples.values()) )
    print(data_loader.nums_examples)


    encoder = Encoder(vocab_size=len(tokenizers["zh"]), pad_token=tokenizers["zh"].pad_token_id).to(device)
    dec_model = Decoder

    dec_params = {"ar": {"vocab_size": len(tokenizers["ar"]), "pad_token": tokenizers["ar"].pad_token_id}}

   
    _, decoders = load_models(None, None, [], dec_model, dec_params, ["ar"], "./checkpoints/diff_families/fourth")
    
    for k, v in decoders.items():
        decoders[k] = v.to(device)

    evaluator = ToyEvaluator(tokenizers, "These are human rights")

    train_conf = SimpleTrainConfig(data_loader, nn.CrossEntropyLoss(), eval_inter=1000, decode_tokenizers=tokenizers, num_epochs=4, evaluator=None, save_models=True, save_prefix="./checkpoints/diff_families/added3", save_inter=10000, log_loss=True, log_file = "./newenc_loss_log3.txt")

    enc_opt = opt.Adam(encoder.parameters(), lr=0.0001)

    new_encoder(train_conf, encoder, enc_opt, decoders, "zh")

    # 4 x 20k batches



def manual_testing():

    dec_params = {"en": {"vocab_size": len(tokenizers["en"]), "pad_token": tokenizers["en"].pad_token_id}}

    enc_params = {"en": {"vocab_size": len(tokenizers["en"]), "pad_token": tokenizers["en"].pad_token_id}}


    encoders, decoders = load_models(Encoder, enc_params, ["en"], Decoder, dec_params, ["en"], "./checkpoints/diff_families/first") 

    eval_model = EvalModel(encoders["en"], decoders["en"], tokenizers["en"], tokenizers["en"])


    print(eval_model.translate("The missile knows where it is because it knows where it isn't. By subtracting where it is from where it isn't, it can know where it is going."))






def continue_train_diff_families():
    data = load_un_pc([("ar", "en"), ("ar", "ru"), ("en", "ru")], tokenizers, 3800000, start_item=2900000, self_map=False)

    data_loader = DataLoader(data, 16)

    print(sum(data_loader.nums_examples.values()) )
    print(data_loader.nums_examples)


    enc_model = Encoder
    dec_model = Decoder

    dec_params = {l: {"vocab_size": len(tokenizers[l]), "pad_token": tokenizers[l].pad_token_id} for l in data_loader.langs}

    enc_params = {l: {"vocab_size": len(tokenizers[l]), "pad_token": tokenizers[l].pad_token_id} for l in data_loader.langs}



    evaluator = ToyEvaluator(tokenizers, "The missile knows where it is.")

    train_conf = TrainConfig(data_loader, enc_model, enc_params, dec_model, dec_params, opt.Adam,
                             nn.CrossEntropyLoss(), enc_opt_params={"lr": 0.0001}, dec_opt_params={"lr": 0.0001},
                             eval_inter=1000, decode_tokenizers=tokenizers, num_epochs=2000, load_model=True,
                             load_prefix="./checkpoints/diff_families/third", evaluator=evaluator,
                             save_models=True, save_prefix="./checkpoints/diff_families/fourth",
                             save_inter=10000, log_loss=True, log_file = "./loss_log.txt")


    print("PARAMETERS: ", count_parameters(dec_model, dec_params["en"]))

    train(train_conf)

    # 23-29  40k batches
    # 29-38  60k batches




def continue_new_enc_one_pair():
    data = load_un_pc([("ar", "zh")], tokenizers, start_item=2900000, self_map=False, do_reverse=False, swap_pairs=True)

    data_loader = DataLoader(data, 16)

    print(sum(data_loader.nums_examples.values()) )
    print(data_loader.nums_examples)


    encoder = Encoder(vocab_size=len(tokenizers["zh"]), pad_token=tokenizers["zh"].pad_token_id).to(device)
    dec_model = Decoder

    dec_params = {"ar": {"vocab_size": len(tokenizers["ar"]), "pad_token": tokenizers["ar"].pad_token_id}}

   
    _, decoders = load_models(None, None, [], dec_model, dec_params, ["ar"], "./checkpoints/diff_families/second")
    
    for k, v in decoders.items():
        decoders[k] = v.to(device)

    evaluator = ToyEvaluator(tokenizers, "These are human rights")

    train_conf = SimpleTrainConfig(data_loader, nn.CrossEntropyLoss(), eval_inter=1000, decode_tokenizers=tokenizers, num_epochs=2000, evaluator=None, save_models=True, save_prefix="./checkpoints/diff_families/added", save_inter=10000, log_loss=True, log_file = "./newenc_loss_log.txt")

    enc_opt = opt.Adam(encoder.parameters(), lr=0.0001)

    new_encoder(train_conf, encoder, enc_opt, decoders, "zh")






def eval_translate():

    eval_data = load_un_pc([("en", "es")], tokenizers, 5010000, start_item=5000000,   self_map=False, do_reverse=False, swap_pairs=True)


    dec_params = {"en": {"vocab_size": len(tokenizers["en"]), "pad_token": tokenizers["en"].pad_token_id}}

    enc_params = {"es": {"vocab_size": len(tokenizers["es"]), "pad_token": tokenizers["es"].pad_token_id}}


    encoders, decoders = load_models(Encoder, enc_params, ["es"], Decoder, dec_params, ["en"],  "./checkpoints/sim_2epoch/sim") 

    eval_model = EvalModel(encoders["es"], decoders["en"], tokenizers["es"], tokenizers["en"])

    es, en = eval_data[("es", "en")]

    es = tokenizers["es"].batch_decode(es)
    es = [s.split("</s>")[0] for s in es]

    en = tokenizers["en"].batch_decode(en)
    en = [s.split("</s>")[0] for s in en]

    with open("translations_es_en_sim.txt", "w") as f:
        t1 = []
        for s in tqdm(es):
            translation = eval_model.translate(s)
            t1.append(translation)
            f.write(translation + "\n")

        
    t2 = en

    print(t1)
    print(t2)

    bleu = eval_bleu(t1, t2, smooth_method=7)

    print(bleu)




def eval_metrics():

    eval_data = load_un_pc([("en", "es")], tokenizers, 5010000, start_item=5000000,   self_map=False, do_reverse=False, swap_pairs=True)




    zh, en = eval_data[("es", "en")]

    zh = tokenizers["es"].batch_decode(zh)
    zh = [s.split("</s>")[0] for s in zh]

    en = tokenizers["en"].batch_decode(en)
    en = [s.split("</s>")[0] for s in en]

    with open("translations_es_en_sim.txt", "r") as f:
        t1 = f.readlines()
        
    t2 = en

    bleu = eval_bleu(t1, t2, smooth_method=7)
    rouge = eval_rouge(t1, t2)
    ball = [eval_bleu(t1, t2, smooth_method=i) for i in range(8)]
   # meteor = eval_meteor(t1, t2)

    print(bleu)
    print(rouge)
    print(ball)
    #print(meteor)










#new_enc_one_pair()
#eval_translate()
eval_metrics()
#continue_train_diff_families()





