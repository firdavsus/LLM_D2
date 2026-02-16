"""
info

### ATTENTION ###

Q - what I am looking for
K - what I mean
V - what I am
! biases == False

Attn = softmax((QK^T) / sqrt(model_dim)) * V
! before multiplying by V, apply masking

softmax - each row (horizontal) == 1

### EMBEDDINGS ###

Embedding[i, 2k] = sin(position / (10000^(2k / d_model)))
Embedding[i, 2k+1] = cos(position / (10000^(2k / d_model)))

### FEEDFORWARD ###

FeedForward - dropout after activation and at the end

## BLOCK ##

1. Add & Norm
2. Attn
3. Add & Norm
4. FeedForward

skip connection + pre-norm + appropriate dropout

## TRANSFORMER ##

!most important pad_mask integrated
ml_head  weights tied to tok_emb

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from transformers.modeling_outputs import CausalLMOutput

# CONFIG
class Config():
    def __init__(self):
        self.vocab_size = 50277
        self.model_dim = 768
        self.max_seq_len = 1024
        self.num_heads = 12
        self.dropout = 0.05
        self.dropout_fn = 0.02
        self.exp_dim = 4
        self.layers = 20

        self.batch_size = 64
        self.num_train_epochs = 50
        self.learning_rate=3e-4
        self.grad_acc=4

# SINUSOIDAL POSITION EMBEDDINGS
def get_sin_emb(config, n=10_000.0):
    positions = torch.arange(0, config.max_seq_len).unsqueeze_(1)
    emb = torch.zeros(config.max_seq_len, config.model_dim)

    denominators = torch.pow(n, 2*torch.arange(0, config.model_dim//2)/config.model_dim)
    emb[:, 0::2] = torch.sin(positions/denominators)
    emb[:, 1::2] = torch.cos(positions/denominators)

    return emb

# EMBEDDINGS
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.model_dim)
        self.register_buffer("pos_emb", get_sin_emb(config))
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=config.model_dim ** -0.5)

    def forward(self, x):
        B, T = x.shape
        return self.tok_emb(x) + self.pos_emb[:T, :]


# ATTENTION
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.K = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.V = nn.Linear(config.model_dim, config.model_dim, bias=False)

        self.out = nn.Linear(config.model_dim, config.model_dim, bias=False)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1)
        )

        self.dropout = nn.Dropout(config.dropout)

        self.model_dim = config.model_dim
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads

        std = config.model_dim ** -0.5
        init.normal_(self.Q.weight, mean=0.0, std=std)
        init.normal_(self.K.weight, mean=0.0, std=std)
        init.normal_(self.V.weight, mean=0.0, std=std)
        init.xavier_uniform_(self.out.weight, gain=1.0)

        init.xavier_uniform_(self.out.weight, gain=1.0)
        if self.out.bias is not None:
            init.zeros_(self.out.bias)


    def forward(self, x, pad_mask=None):
        B, T, C = x.shape # C is dimention_out 

        Q = self.Q(x) # (B, T, C)
        K = self.K(x)
        V = self.V(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2) # (B, num_heads, T, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_scores = Q @ K.transpose(2, 3)
    
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        mask_bool = self.mask[:T, :T].unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        attn_scores = attn_scores.masked_fill(mask_bool.bool(), float('-1e9'))
     
        if pad_mask is not None:
            pad_mask = pad_mask[:, None, None, :].bool()
            attn_scores = attn_scores.masked_fill(~pad_mask, float('-1e9'))
     
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ V).transpose(1, 2)  # Shape: (B, T, num_heads, head_dim)

        context_vec = context_vec.contiguous().view(B, T, self.model_dim)
        context_vec = self.out(context_vec)

        return context_vec

# SWIGELU
class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()
    
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

# FEEDFORWARD
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.exp_dim * config.model_dim
        self.fc1 = nn.Linear(config.model_dim, hidden_dim)
        self.act = SwiGLU()
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, config.model_dim)
        self.dropout2 = nn.Dropout(config.dropout)

        # Initialization
        init.xavier_uniform_(self.fc1.weight, gain=math.sqrt(2.0))
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight, gain=1.0)
        init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.dropout2(self.fc2(self.dropout1(self.act(self.fc1(x)))))

        
# BLOCK
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.model_dim)
        self.attn = MultiHeadAttention(config)
        self.drop1 = nn.Dropout(config.dropout)

        self.ln2 = nn.RMSNorm(config.model_dim)
        self.ffn = FeedForward(config)
        self.drop2 = nn.Dropout(config.dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop1(self.attn(self.ln1(x), pad_mask))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x

# TRANSFORMER
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = Embeddings(config)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.ln = nn.RMSNorm(config.model_dim)
        self.out = nn.Linear(config.model_dim, config.vocab_size)
        self.out.weight = self.emb.tok_emb.weight
        self.drop = nn.Dropout(config.dropout_fn)
        self.config = config

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.emb(input_ids)

        pad_mask = attention_mask if attention_mask is not None else None
        for layer in self.layers:
            x = layer(x, pad_mask)

        x = self.ln(x)
        x = self.drop(x)
        logits = self.out(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits)
