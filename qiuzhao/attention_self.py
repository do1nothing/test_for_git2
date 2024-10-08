import torch
import torch.nn as nn
import copy

class mulitihead_attention(nn.Module):
    def __init__(self,hidden_size,heads):
        super(mulitihead_attention,self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        assert hidden_size%heads==0
        self.head_dim = hidden_size // heads
        
        self.q = nn.Linear(hidden_size,hidden_size,bias=False)
        self.k = nn.Linear(hidden_size,hidden_size,bias=False)
        self.v = nn.Linear(hidden_size,hidden_size,bias=False)
        
        self.fc = nn.Linear(hidden_size,hidden_size,bias=False)
        
    def forward(self, query, key, value, mask=None):
        bs = query.shape[0]
        seq_size = query.shape[1]
        
        q = self.q(query).view(bs,seq_size,self.heads, self.head_dim)
        k = self.k(key).view(bs,seq_size,self.heads, self.head_dim)
        v = self.v(value).view(bs,seq_size,self.heads, self.head_dim)

        scores = torch.einsum("bqhd,bkhd->bhqk",q,k)
        scores = (scores / self.hidden_size ** (1/2)).softmax(dim = -1)
        
        if mask is not None:
            scores = scores.make_fill(mask == 0,-1e10)
            
        attention = torch.einsum("bhqk,bkhd->bqhd",scores,v)
        attention = self.fc(attention.reshape(bs,seq_size,self.hidden_size))
        
        return attention
        
        
if __name__ == "__main__":
    bs, seq_size, hidden_size = 128, 32, 16
    heads = 8
    mask = None
    attention = mulitihead_attention(hidden_size, heads)
    [query, key, value] = [torch.randn(bs, seq_size, hidden_size)]*3
    scores = attention(query, key, value, mask)
    print(scores.shape)