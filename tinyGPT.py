"""
tinyGPT.py
==========================================
A compact, educational decoder-only transformer, written in the
spirit of Andrej Karpathy's minGPT/nanoGPT/nanochat. The entire language model lives
in this one file:

    - character-level tokenizer
    - token + positional embeddings
    - multi-head *causal* self-attention
    - feed-forward MLP blocks with residual connections + LayerNorm
    - autoregressive sampling (generate)
    - a tiny self-contained training loop (no downloads needed)

Run it:
    pip install torch
    python nano_gpt.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# model config
@dataclass
class GPTConfig:
    block_size: int = 64      # max context length (how far back a token can attend)
    vocab_size: int = 65      # number of distinct tokens; set from the dataset
    n_layer: int = 4          # number of stacked Transformer blocks
    n_head: int = 4           # number of attention heads per block
    n_embd: int = 128         # width of the residual stream / embeddings
    dropout: float = 0.1


# causal self attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # one big linear projects the input into queries, keys and values at once
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # output projection
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # a lower-triangular mask so position t can only attend to positions <= t.
        # registered as a buffer so it moves with .to(device) but isn't a parameter.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()                  # batch, time (tokens), channels (n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        # split the channel dim into heads -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # scaled dot-product attention: how much should each token read from each other?
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)   # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)        # turn scores into a probability mix
        att = self.attn_dropout(att)

        y = att @ v                         # weighted sum of values -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-merge the heads
        y = self.resid_dropout(self.c_proj(y))
        return y


# mlp
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # expand 4x
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # project back
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# transformer
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # communicate: tokens mix information
        x = x + self.mlp(self.ln_2(x))    # compute: process each token
        return x


# gpt
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)  # "what" a token is
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)    # "where" it sits
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: the input embedding table and the output projection
        # share the same weights (saves parameters and tends to help).
        self.token_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "sequence longer than block_size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok = self.token_emb(idx)           # (B, T, n_embd)
        pos = self.pos_emb(pos)             # (T, n_embd), broadcast over batch
        x = self.drop(tok + pos)            # the residual stream begins here
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)            # (B, T, vocab_size): score for next token

        loss = None
        if targets is not None:
            # next-token prediction = classification over the vocabulary
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend a sequence one token at a time."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]      # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature          # only the last position matters
            if top_k is not None:                            # optional top-k filtering
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample, don't argmax
            idx = torch.cat((idx, next_id), dim=1)
        return idx


# demo, dependency-free
def demo():
    # a minuscule corpus baked in, so there is nothing to download
    text = "to be or not to be that is the question " * 300

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)

    data = torch.tensor(encode(text), dtype=torch.long)

    config = GPTConfig(vocab_size=len(chars), block_size=32,
                       n_layer=3, n_head=4, n_embd=96)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e3:.1f}K  "
          f"| device: {device}")

    def get_batch(batch_size=32):
        ix = torch.randint(len(data) - config.block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    for step in range(500):
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    model.eval()
    start = torch.tensor([encode("to be")], dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=80, temperature=0.8, top_k=10)
    print("\nSample:\n" + decode(out[0].tolist()))


if __name__ == "__main__":
    torch.manual_seed(1337)
    demo()
