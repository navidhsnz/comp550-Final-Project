# %%
!pip install torchtext==0.6.0

# %%
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm
!python -m spacy download fr_core_news_sm
!python -m spacy download xx_sent_ud_sm

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
import random
import math
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

dataset_path="datasets"

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
def spacy_tokenize(spacy_lang, reverse = False):
    ''' (spacy.lang.xx, bool) -> (func)
    tokenizes a sentence using spacy.
    '''
    if reverse:
        return lambda sent: [tok.text for tok in spacy_lang.tokenizer(sent)][::-1]
    else:
        return lambda sent: [tok.text for tok in spacy_lang.tokenizer(sent)]


def field(spacy_lang, reverse=False):
    ''' (spacy.lang.xx, bool) -> (torchtext.data.Field)
    '''
    return Field(tokenize = spacy_tokenize(spacy_lang, reverse = reverse),
                 init_token = '<sos>',
                 eos_token = '<eos>',
                 lower = True)


# %%
# Loading languages from spacy
spacy_de = spacy.load('de_core_news_sm')
spacy_fr = spacy.load('fr_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')
spacy_cs = spacy.load('xx_sent_ud_sm')

# Define source and target fields
SRC_de = field(spacy_de)
SRC_fr = field(spacy_fr)
SRC_en = field(spacy_en)
SRC_cs = field(spacy_cs)

# Loading and splitting the data into train, validation and test sets
train_data_en_to_de, valid_data_en_to_de, test_data_en_to_de = Multi30k.splits(exts = ('.en', '.de'), fields = (SRC_en,  SRC_de), path=dataset_path)
train_data_en_to_fr, valid_data_en_to_fr, test_data_en_to_fr = Multi30k.splits(exts = ('.en', '.fr'), fields = (SRC_en,  SRC_fr), path=dataset_path)
train_data_en_to_cs, valid_data_en_to_cs, test_data_en_to_cs = Multi30k.splits(exts = ('.en', '.cs'), fields = (SRC_en,  SRC_cs), path=dataset_path)
train_data_en_to_en, valid_data_en_to_en, test_data_en_to_en = Multi30k.splits(exts = ('.en', '.en'), fields = (SRC_en,  SRC_en), path=dataset_path)

# %%
print(f"Number of training examples: {len(train_data_en_to_de.examples)}")
print(f"Number of validation examples: {len(valid_data_en_to_de.examples)}")
print(f"Number of testing examples: {len(test_data_en_to_de.examples)}")

# %%
#test dataset
i=0
print("German:",' '.join(test_data_en_to_de.examples[i].trg))
print("French:",' '.join(test_data_en_to_fr.examples[i].trg))
print("Czech:",' '.join(test_data_en_to_cs.examples[i].trg))
print("English:",' '.join(test_data_en_to_en.examples[i].trg))

# %%
SRC_de.build_vocab(train_data_en_to_de, min_freq = 2)
SRC_fr.build_vocab(train_data_en_to_fr, min_freq = 2)
SRC_cs.build_vocab(train_data_en_to_cs, min_freq = 2)
SRC_en.build_vocab(train_data_en_to_en, min_freq = 2)
#TRG.build_vocab(train_data_en_to_en, min_freq = 2)

print(f"Unique tokens in source (de) vocabulary: {len(SRC_de.vocab)}")
print(f"Unique tokens in source (fr) vocabulary: {len(SRC_fr.vocab)}")
print(f"Unique tokens in source (cs) vocabulary: {len(SRC_cs.vocab)}")
print(f"Unique tokens in source (en) vocabulary: {len(SRC_en.vocab)}")

# %%
BATCH_SIZE = 200

train_iterator_en_to_de, valid_iterator_en_to_de, test_iterator_en_to_de = BucketIterator.splits(
    (train_data_en_to_de, test_data_en_to_de, valid_data_en_to_de),
    batch_size = BATCH_SIZE,
    device = device)

train_iterator_en_to_fr, valid_iterator_en_to_fr, test_iterator_en_to_fr = BucketIterator.splits(
    (train_data_en_to_fr, test_data_en_to_fr, valid_data_en_to_fr),
    batch_size = BATCH_SIZE,
    device = device)

train_iterator_en_to_cs, valid_iterator_en_to_cs, test_iterator_en_to_cs = BucketIterator.splits(
    (train_data_en_to_cs, test_data_en_to_cs, valid_data_en_to_cs),
    batch_size = BATCH_SIZE,
    device = device)

train_iterator_en_to_en, valid_iterator_en_to_en, test_iterator_en_to_en = BucketIterator.splits(
    (train_data_en_to_en, test_data_en_to_en, valid_data_en_to_en),
    batch_size = BATCH_SIZE,
    device = device)

# %%
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# %%
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0,:]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output

            #teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

# %%
INPUT_DIM = len(SRC_en.vocab)
OUTPUT_DIM = max(len(SRC_de.vocab),len(SRC_fr.vocab))
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 1024
N_LAYERS = 3
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc_de = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
enc_fr = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)


model_en_de = Seq2Seq(enc_de, dec, device).to(device)
model_en_fr = Seq2Seq(enc_fr, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model_en_de.apply(init_weights)
model_en_fr.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The models have {count_parameters(model_en_de):,} and {count_parameters(model_en_fr):,} trainable parameters')

optimizer_en_de = optim.Adam(model_en_de.parameters())
optimizer_en_fr = optim.Adam(model_en_fr.parameters())

TRG_PAD_IDX = SRC_en.vocab.stoi[SRC_en.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX) #ignore_index = TRG_PAD_IDX


def train(model, training_iterator, optimizer, criterion, clip):

    model.train()
    epoch_loss = 0

    for i , data_itr in enumerate(training_iterator):
        
        src = data_itr.src
        trg = data_itr.trg
        optimizer.zero_grad()
        # print(src.shape)
        # print(trg.shape)
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        # print(output.shape)
        # print(trg.shape)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(training_iterator)

def evaluate(model, iterator, criterion):

    model.eval()
    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0) #turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 30
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    print("started")
    start_time = time.time()
    
    #rand_indx = random.randint(0, len(training_iterator_list)-1)
    
    train_loss_fr = train(model_en_fr, train_iterator_en_to_fr, optimizer_en_fr, criterion, CLIP)
    valid_loss_fr = evaluate(model_en_fr, valid_iterator_en_to_fr, criterion)

    train_loss_de = train(model_en_de, train_iterator_en_to_de, optimizer_en_de, criterion, CLIP)
    valid_loss_de = evaluate(model_en_de, valid_iterator_en_to_de, criterion)

    train_loss = 1/2 * (train_loss_fr+train_loss_de)
    valid_loss = 1/2 * (valid_loss_fr+valid_loss_de)

    print("ended")

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model_en_de.state_dict(), 'saved_model_en_de.pt')
        torch.save(model_en_fr.state_dict(), 'saved_model_en_fr.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# %%
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()

    tokenized_sentence = src_field.tokenize(sentence)
    tokenized_sentence = [src_field.init_token] + tokenized_sentence + [src_field.eos_token]
    numericalized_sentence = [src_field.vocab.stoi[token] for token in tokenized_sentence]
    input_tensor = torch.LongTensor(numericalized_sentence).unsqueeze(1).to(device)
    #input_length = torch.tensor([len(numericalized_sentence)])
    #print(input_tensor)
    with torch.no_grad():
        encoder_hidden, encoder_cell = model.encoder(input_tensor)
    #print(encoder_hidden)
    trg_input = torch.tensor([trg_field.vocab.stoi[trg_field.init_token]]).to(device)
    output_words = []

    for _ in range(max_len):
        # Forward pass through the decoder
        with torch.no_grad():
            output, encoder_hidden, encoder_cell = model.decoder(trg_input, encoder_hidden, encoder_cell)
        pred_token = output.argmax(1).item()

        output_words.append(trg_field.vocab.itos[pred_token])
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

        trg_input = torch.tensor([pred_token]).to(device)

    translated_sentence = ' '.join(output_words)

    return translated_sentence

input_sentence = "woman"
translation_fr = translate_sentence(input_sentence, SRC_en, SRC_fr, model_en_fr, device)
translation_de = translate_sentence(input_sentence, SRC_en, SRC_de, model_en_de, device)
print(f"Input: {input_sentence} \
      \nTranslation to German: {translation_de}  \
      \nTranslation to French: {translation_fr}")


# %%
