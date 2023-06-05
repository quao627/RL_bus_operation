import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
import copy
from Sublayers import FeedForward, MultiHeadAttention, Norm


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class OutLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        
    def forward(self, x):
        x2 = self.norm_1(x)
        x = x[..., 0, :] + self.dropout_1(self.attn(x2,x2,x2)[..., 0, :])
        return x

class BusEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, N=5, heads=8, dropout=0.1):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.out = OutLayer(d_model, heads, dropout=dropout)
        self.norm = Norm(d_model)

    def forward(self, src):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x)
        x = self.out(x)
        return self.norm(x)

if __name__ == '__main__':
    model = BusEncoder(8, 32)
    obs = torch.zeros(1, 12, 8)
    print(model(obs))
    # print(model(obs).size())