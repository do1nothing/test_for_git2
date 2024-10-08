import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        # Linear layers for query, key, and value projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output linear layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        query = self.query_proj(query)  # (batch_size, query_len, embed_dim)
        key = self.key_proj(key)        # (batch_size, key_len, embed_dim)
        value = self.value_proj(value)  # (batch_size, value_len, embed_dim)
        
        # Split into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        # attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / self.head_dim ** 0.5
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # attention_output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, query_len, head_dim)
        attention_output = torch.einsum("bhqk,bkhd->bqhd", attention_probs, value)
        # Concatenate heads and project
        # attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, -1, self.embed_dim)
        
        output = self.out_proj(attention_output)
        return output

if __name__ == "__main__":
    batch_size = 2
    query_len = 5  # 查询序列长度
    key_value_len = 7  # 键-值序列长度
    embed_dim = 32  # 嵌入维度
    num_heads = 4  # 多头数

    query = torch.randn(batch_size, query_len, embed_dim)
    key = torch.randn(batch_size, key_value_len, embed_dim)
    value = torch.randn(batch_size, key_value_len, embed_dim)

    cross_attention = MultiHeadCrossAttention(embed_dim, num_heads)
    output = cross_attention(query, key, value)
    print(output.shape)  # 输出 (batch_size, query_len, embed_dim)