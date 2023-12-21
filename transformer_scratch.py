

from os import device_encoding
from fsspec.utils import tokenize
import torch
from torch._dynamo import optimize
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer


DEVICE = "cuda"



class AttentionHead(nn.Module):

    def __init__(self, head_size, num_embed, block_size, dropout) -> None:
        super().__init__()

        # What if we make these nonlinear?? (like neural net)
        self.key = nn.Linear(num_embed, head_size, bias=False)   
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)


        # I don't really get this
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)


        # I don't really get this
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out



class MultiHeadAttention(nn.Module):


    def __init__(self, num_heads, head_size, num_embed, block_size, dropout) -> None:
        super().__init__()

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    head_size=head_size,
                    num_embed=num_embed,
                    block_size=block_size,
                    dropout=dropout
                )
                for _ in range(num_heads)
            ]
        )

        self.proj = nn.Linear(num_embed, num_embed)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        out = self.proj(out)
        return out





class FeedForward(nn.Module):

    def __init__(self, num_embed) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            # Why 4?? try other things???
            nn.Linear(num_embed, 4*num_embed),
            nn.ReLU(),
            nn.Linear(4*num_embed, num_embed)
        )


    def forward(self, x):
        return self.net(x)



class TransformerBlock(nn.Module):

    def __init__(self, num_heads, block_size, num_embed, dropout) -> None:
        super().__init__()
        
        head_size = num_embed // num_heads

        self.sa = MultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            num_embed=num_embed,
            block_size=block_size,
            dropout=dropout
        )

        self.ffwd = FeedForward(num_embed=num_embed)

        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)


    def forward(self, x):

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x




class Transformer(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.vocab_size = kwargs.get("vocab_size")
        self.num_embed = kwargs.get("num_embed")
        self.block_size = kwargs.get("block_size")
        self.num_heads = kwargs.get("num_heads")
        self.num_layers = kwargs.get("num_layers")
        self.dropout = kwargs.get("dropout")

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.num_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, self.num_embed)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    num_heads=self.num_heads,
                    block_size=self.block_size,
                    num_embed=self.num_embed,
                    dropout=self.dropout
                )
                for _ in range(self.num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(self.num_embed)
        self.lm_head = nn.Linear(self.num_embed, self.vocab_size)



    def forward(self, idx, targets=None):

        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        posit_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))

        x = token_emb + posit_emb

        x = self.blocks(x)

        logits = self.lm_head(x)


        if targets != None:

            B, T, C = logits.shape
            logits = torch.reshape(logits, (B*T, C))
            targets = torch.reshape(targets, (B*T,))
            loss = F.cross_entropy(logits, targets)

        else:
            loss = None

        return logits, loss


    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):

        for _ in range(max_new_tokens):

            idx_crop = idx[:, -block_size:]

            logits, loss = self.forward(idx_crop)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        print(probs)

        return idx
          
exit()

if __name__ == "__main__":

    import data_utils

    BATCH_SIZE = 128
    BLOCK_SIZE = 128
    MAX_ITER = 50000
    EVAL_INTER = 100
    SAVE_INTER = 3000
    LEARNING_RATE = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_HEAD = 8
    NUM_EMBED = NUM_HEAD * 128
    NUM_LAYER = 6
    DROPOUT = 0.2

#DEVICE="cpu"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size


    model = Transformer(
        vocab_size=vocab_size,
        num_embed=NUM_EMBED,
        block_size=BLOCK_SIZE,
        num_heads=NUM_HEAD,
        num_layers=NUM_LAYER,
        dropout=DROPOUT,
    )


    print(sum(p.numel() for p in model.parameters()), "parameters")



    model.load_state_dict(torch.load("checkpoints/gpt_scratch4000.pt"))



    print("Moving model to gpu")
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)



    test_prompt = ("In order to test the value")



    for step in range(4000, MAX_ITER):


        x, y = data_utils.get_batch(tokenizer, BLOCK_SIZE, BATCH_SIZE)

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits, loss = model.forward(x, y)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()



        if step % EVAL_INTER == 0:
            print(step)
            print(loss)

            with torch.no_grad():
                test_tokens = tokenizer.tokenize(test_prompt) 
                test_ids = tokenizer.convert_tokens_to_ids(test_tokens)
                test_ids = torch.tensor(test_ids).unsqueeze(0).to(DEVICE)

                test_out = model.generate(test_ids, 64, BLOCK_SIZE)

                gen_text = tokenizer.batch_decode(test_out)[0]
                print(gen_text)



        if step % SAVE_INTER == 0:
            torch.save(model.state_dict(), "checkpoints/gpt_scratch" + str(step) + ".pt")

            

    



