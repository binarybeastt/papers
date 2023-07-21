# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import torchtext
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
print(torch.__version__)

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x):
        out = self.embed(x)
        return out
        
class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_model_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_model_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_model_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_model_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_model_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_model_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x
    

class MultiHeadAttention(nn.Module):
    """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
    def __init__(self, embed_dim=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim/self.n_heads)
        
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim, self.embed_dim)
        
    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        
        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        # Query dimension can change in the decoder during inference.
        # So we can't take a general seq_length.
        seq_length_query = query.size(1)
        
        # Reshape key tensor to (batch_size x sequence_length x n_heads x single_head_dim)
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        
        k = self.key_matrix(key) # Apply linear transformation to keys (batch_size x seq_len x n_heads x single_head_dim)
        q = self.query_matrix(query) # Apply linear transformation to query (batch_size x seq_len x n_heads x single_head_dim)
        v = self.value_matrix(value) # Apply linear transformation to values (batch_size x seq_len x n_heads x single_head_dim)
        
        q = q.transpose(1,2) # Transpose queries to (batch_size, n_heads, seq_len_query, single_head_dim)
        k = k.transpose(1,2) # Transpose keys to (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2) # Transpose values to (batch_size, n_heads, seq_len, single_head_dim)
        
        # Compute attention scores (dot product of queries and keys)
        product = torch.matmul(q, k.transpose(-1,-2)) # (batch_size x n_heads x seq_len_query x seq_len)
        
        # Fill positions where mask is 0 with a large negative value to ignore them during softmax
        if mask is not None:
            product = product.masked_fill(mask==0, float('-1e20'))
        
        # Normalize attention scores with sqrt(d_k) for more stable gradients
        product = product/math.sqrt(self.single_head_dim)
        # Apply softmax to obtain attention weights
        scores = F.softmax(product, dim=-1)
        # Weighted sum of values using attention weights to get the final attention output
        scores = torch.matmul(scores, v) # (batch_size x n_heads x seq_len_query x single_head_dim)
        # Concatenate attention outputs from all heads
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query, self.single_head_dim * self.n_heads)
        # Linear layer to bring the output back to the original embedding dimension
        output = self.out(concat)  # (batch_size x seq_len_query x embed_dim)

        return output
    
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()
        
        """
        Args:
        embed_dim: dimension of the embedding
        expansion_factor: fator ehich determines output dimension of linear layer
        n_heads: number of attention heads  
        """
        
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor*embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor*embed_dim, embed_dim)
        )
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, key, query, value):
        """
        Args:
        key: key vector
        query: query vector
        value: value vector
        norm2_out: output of transformer block
        
        """
        
        attention_out = self.attention(key, query, value)
        attention_residual_out = attention_out + value
        norm1_out = self.dropout1(self.norm1(attention_residual_out))
        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) #32x10x512

        return norm2_out
    
class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerEncoder,self).__init__()
        
        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        
        self.layers = TransformerBlock([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
    
    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out,out,out)

        return out  #32x10x512
    

        
        
    
    