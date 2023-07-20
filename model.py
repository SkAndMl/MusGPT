import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

class Embedding(nn.Module):

    def __init__(self,
                 config,
                 vocab_size):
        """
            Embedding generates learnable representation of an input sequence which encodes
            contextual, semantic meaning for each word.
            Params:
                d_model(int): specifies the embedding dimension for each token/word
                vocab_size(int): number of embeddings that would be needed. # of unique words
                max_seq_len(int): the maximum sequence length of an input sequence. Used for generation positional encoding
                dropout(float): probability of dropout applied on the final embedding output
        """
        
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size,
                                                  embedding_dim=config["d_model"])
        self.position_embedding_table = nn.Embedding(num_embeddings=config["context_length"],
                                                     embedding_dim=config["d_model"])
        self.dropout = nn.Dropout(p=config["dropout"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x => [B, S]
        B, S = x.shape
        token_emb = self.token_embedding_table(x) # [B, S, D]

        pos_emb = self.position_embedding_table(torch.arange(S, device=device)).unsqueeze(0) # [1, S, D]
        out = self.dropout(token_emb+pos_emb)
        return self.dropout(out)
    


class AttentionHead(nn.Module):

    def __init__(self,
                 config) -> None:
        
        super().__init__()

        self.d_model = config["d_model"]
        self.head_dim = config["head_dim"]

        self.query = nn.Linear(self.d_model, self.head_dim)
        self.key = nn.Linear(self.d_model, self.head_dim)
        self.value = nn.Linear(self.d_model, self.head_dim)
        self.dropout = nn.Dropout(p=config["dropout"])

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask=None) -> torch.Tensor:
        
        # query => [B, Q, D]
        # key => [B, K, D]
        # value => [B, K, D]

        q = self.query(query) # B, Q, HEAD_DIM 
        k = self.key(key) # B, K, HEAD_DIM
        v = self.value(value) # B, K, HEAD_DIM

        weights = q @ k.transpose(1, 2) # B, Q, K
        if mask is not None:
            weights = weights.masked_fill(mask==0, value=float("-inf"))
        weights = F.softmax(weights/math.sqrt(self.head_dim), dim=-1)
        out = weights @ v # [B, Q, K] x [B, K, HEAD_DIM] => [B, Q, HEAD_DIM]
        return self.dropout(out)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 config) -> None:
         
         super().__init__()
         self.sa_heads = nn.ModuleList([AttentionHead(config) for _ in range(config["n_heads"])])
         self.proj = nn.Linear(config["d_model"], config["d_model"])
         self.dropout = nn.Dropout(p=config["dropout"])
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask=None) -> torch.Tensor:
        
        out = torch.cat([h(query, key, value, mask) for h in self.sa_heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):

    def __init__(self,
                 config):
        
        super().__init__()
        d_model = config["d_model"]
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(p=config["dropout"])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.net(x)
        return x


class GPTDecoderBlock(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln_1 = nn.LayerNorm(normalized_shape=config["d_model"])
        self.ln_2 = nn.LayerNorm(normalized_shape=config["d_model"])

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:

        x = x + self.mha(self.ln_1(x), self.ln_1(x), self.ln_1(x), mask)
        x = x + self.ff(self.ln_2(x))
        return x

class GPTDecoder(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        self.blocks = nn.ModuleList([GPTDecoderBlock(config) for _ in range(config["n_decoders"])])
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:

        for block in self.blocks:
            x = block(x, mask)
        return x
    
class PoemGPT(nn.Module):

    def __init__(self, config, vocab_size) -> None:
        
        super().__init__()
        self.context_length = config["context_length"]
        self.embedding = Embedding(config, vocab_size)
        self.gpt = GPTDecoder(config)
        self.lm_head = nn.Linear(config["d_model"], vocab_size)
    
    def forward(self,
                x: torch.Tensor,
                targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, S = x.shape
        # x -> [B, S], targets -> [B, S]
        x = self.embedding(x) # B, S, D_MODEL
        mask = create_causal_mask(S)
        
        x = self.gpt(x, mask) # B, S, D_MODEL
        logits = self.lm_head(x) # B, S, VOCAB_SIZE

        if targets is None:
            loss = None
        else:
            logits = logits.view(B*S, -1)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    

    def generate(self, x:torch.Tensor=None, max_new_tokens: int=500) -> torch.Tensor:

        if x is None:
            x = torch.zeros((1, 1), dtype=torch.long, device=device) # B, S
        
        for _ in range(max_new_tokens):
            preds, _ = self(x[:, -self.context_length:])# B, S, VOCAB_SIZE
            preds = preds[:, -1, :] # B, VOCAB_SIZE
            probs = F.softmax(preds, dim=-1)
            x_next = torch.multinomial(input=probs, num_samples=1) # B, 1
            x = torch.cat((x, x_next), dim=1) # B, S+1

        return x


def create_causal_mask(sz):
    mask = torch.ones((sz, sz), device=device)
    mask = torch.tril(mask)
    return mask