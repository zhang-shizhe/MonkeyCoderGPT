"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
2) nanoGPT from karpathy:
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size: int = 91
    context_length: int = 256
    num_heads: int = 6
    head_size: int = 64
    emb_dim: int = head_size * num_heads
    num_layers: int = 6
    dropout: float = 0.2


class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # works as a look up table for the probability of the next char for each current char
        self.token_embedding_table = nn.Embedding(
            self.config.vocab_size, self.config.emb_dim
        )
        self.position_embedding_table = nn.Embedding(
            self.config.context_length, self.config.emb_dim
        )
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(self.config.num_layers)]
        )
        self.ln_final = nn.LayerNorm(
            self.config.emb_dim
        )  # the final layer norm before output
        self.lm_head = nn.Linear(self.config.emb_dim, self.config.vocab_size)

    def forward(self, context_idxs, target_idxs=None):
        (
            B,
            T,
        ) = context_idxs.shape  # num of batches; num of total steps in context_length

        # context_idxs, target_idxs are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(context_idxs)  # (B, T, emb_dim)
        position_emb = self.position_embedding_table(
            torch.arange(T, device=self.config.device)
        )  # (T, emb_dim)
        x = token_emb + position_emb  # (B, T, emb_dim)
        x = self.blocks(x)  # (B, T, head_size)
        logits = self.lm_head(
            x
        )  # (B, T, vocab_size), now the feature_dim is vocab_size again

        if target_idxs is None:
            loss = None
        else:
            (
                B,
                T,
                D,
            ) = (
                logits.shape
            )  # num of batches; num of total steps in context_length; num of feature dimension
            logits = logits.view(B * T, D)  # now D == vocab_size == number of classes
            target_idxs = target_idxs.view(B * T)
            loss = F.cross_entropy(logits, target_idxs)

        return logits, loss

    @torch.no_grad()
    def generate(self, context_idxs, max_new_tokens):
        for _ in range(max_new_tokens):
            # trim input
            input_idxs = context_idxs[:, -self.config.context_length:]
            # forward
            logits, loss = self(input_idxs)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, D) tensor for the last step
            probs = F.softmax(logits, dim=-1)  # predicted_label (B, D)

            # sample from the distribution
            # torch.multinomial: Returns a tensor where each row contains num_samples indices
            # sampled from the multinomial probability distribution located in the corresponding row of tensor input.
            pred_idxs = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            context_idxs = torch.cat((context_idxs, pred_idxs), dim=1)  # (B, T+1)
        return context_idxs


class Block(nn.Module):
    """
    a decoder block without cross-attentioin part
    """

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.ln2 = nn.LayerNorm(config.emb_dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MultiHeadAttention(nn.Module):
    """
    multiple heads fo self-attention in parallel
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([Head(config) for _ in range(self.config.num_heads)])
        self.projection = nn.Linear(self.config.emb_dim, self.config.emb_dim)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class Head(nn.Module):
    """
    self-attention with only one head
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.key = nn.Linear(self.config.emb_dim, self.config.head_size, bias=False)
        self.query = nn.Linear(self.config.emb_dim, self.config.head_size, bias=False)
        self.value = nn.Linear(self.config.emb_dim, self.config.head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(self.config.context_length, self.config.context_length)
            ),
        )

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        B, T, D = x.shape
        k = self.key(x)
        q = self.query(x)

        # attention-score
        weight = (
            q @ k.transpose(-2, -1) * D**-0.5
        )  # (B, T, D) @ (B, D, T) ---> (B, T, T)
        # D**-0.5: to relief the influence of large value makes the vector after softmax looks like one-hot vector.

        weight = weight.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B,T,T), the upper-right triangle will be -inf
        weight = F.softmax(weight, dim=-1)  # (B, T, T)
        weight = self.dropout(weight)

        # weighted-aggregation of values based on the attention-score
        v = self.value(x)  # (B, T, D)
        out = weight @ v  # (B, T, T) @ (B, T, D) --------> (B, T, D)

        return out


class FeedForward(nn.Module):
    """
    a simple linear layer with activation in decoder, + projection
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(
                self.config.emb_dim, 4 * self.config.emb_dim
            ),  # the inner dimension is 4 * D, based on the original paper
            nn.ReLU(),
            nn.Linear(4 * self.config.emb_dim, self.config.emb_dim),
            nn.Dropout(self.config.dropout),
        )

    def forward(self, x):
        return self.net(x)
