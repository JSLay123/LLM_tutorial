"""基于transformer搭建一个简单的语言模型，主要用于学习transformer的原理和实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import tiktoken
import pandas as pd
import math

# 超参数
batch_size = 4
context_length = 64 # 长度
d_model = 512        # 维度
num_blocks = 8      # 块数
num_heads = 4       # 头数
head_dim = d_model // num_heads
learnint_rate = 1e-3
dropout = 0.1
max_iters = 1000
eval_interval = 50
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# 获取数据集
if not os.path.exists('sales_textbook.txt'):
  url='https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
  with open('sales_textbook.txt','wb') as f:
    f.write(requests.get(url).content)

with open('sales_textbook.txt','r') as f:
    text=f.read()

enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

tokenized_text=enc.encode(text) # list
# shape: [76923]
tokenized_text =torch.tensor(tokenized_text, dtype=torch.long) # tensor
# print(tokenized_text.shape) # torch.Size([76923])
max_token_value =tokenized_text.max().item()

# 划分训练集和测试集
train_index = int(0.9 * len(tokenized_text))  # 69230
train_data = tokenized_text[:train_index]
valid_data = tokenized_text[train_index:]

data = train_data

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward,self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.ffn(x)

# define scaled dot product attention
class Attention(nn.Module):
    def __init__(self, d_model, head_dim, context_length, dropout):
        super(Attention,self).__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.context_length = context_length
        self.dropout = dropout

        self.key_layer = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.query_layer = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.value_layer = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.dropout_layer = nn.Dropout(self.dropout)

        self.register_buffer("tril", torch.tril(torch.ones((context_length, context_length))))

    def forward(self, x):
        B, T, C = x.shape  # Batch size, Time steps(current context_length), Channels(dimensions)
        assert T <= self.context_length
        assert C == self.d_model
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout_layer(attention_probs)

        out = torch.matmul(attention_probs, v)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_length, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.context_length = context_length
        self.dropout = dropout

        self.heads = nn.ModuleList(
            [Attention(self.d_model, self.head_dim, self.context_length, self.dropout) for _ in range(self.num_heads)]
            )
        self.projection = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        return self.dropout_layer(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, context_length, dropout):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.context_length = context_length
        self.num_heads = num_heads
        self.dropout = dropout

        self.multihead_attention = MultiHeadAttention(self.d_model, self.num_heads, self.context_length, self.dropout)
        self.feed_forward = FeedForward(self.d_model, self.d_model*4, self.dropout)
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)
    
    def forward(self, x):
        x = x + self.multihead_attention(x)
        x = self.layer_norm1(x)
        x = x + self.feed_forward(x)
        x = self.layer_norm2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model//2,)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)    # (max_seq_len,d_model//2) 
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)      # (1,max_seq_len,d_model)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x


class Transformer_Language_Model(nn.Module):
    def __init__(self, d_model, num_heads, num_blocks, context_length, dropout, max_token_value):
        super(Transformer_Language_Model, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.context_length = context_length
        self.dropout = dropout
        self.max_token_value = max_token_value
        # Set up token embedding look-up table
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value + 1, embedding_dim=self.d_model)

        # run all the transformer blocks
        self.positional_encoder = PositionalEncoding(self.d_model, self.context_length)
        self.transformer_blocks = nn.Sequential(
            *([TransformerBlock(d_model=self.d_model, num_heads=self.num_heads, context_length=self.context_length, dropout=self.dropout) 
              for _ in range(self.num_blocks)]+
              [nn.LayerNorm(self.d_model)])
        )
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value+1)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        x = self.positional_encoder(self.token_embedding_lookup_table(idx))
        x = self.transformer_blocks(x)
        logits = self.language_model_out_linear_layer(x)    # [B, T, C]

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
         # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the max size of our positional embeddings table
            idx_crop = idx[:, -self.context_length:]
            # Get predictions
            logits, loss = self(idx_crop)
            # Get the last time step from logits where the dimensions of the logits are (B,T,C)
            logits_last_timestep = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # Sample from the probabilities' distribution.
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # Append the sampled indexes idx_next to idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = Transformer_Language_Model(d_model=d_model, num_heads=num_heads, num_blocks=num_blocks, context_length=context_length, max_token_value=max_token_value, dropout=dropout)
model = model.to(device)

# 获取batch_size条随机句子
def get_batch(split: str, batch_size: int):
    data = train_data if split == 'train' else valid_data
    idxs = torch.randint(0, len(data)-context_length, (batch_size,))
    x_batch = torch.stack([data[i:i+context_length] for i in idxs]).to(device)
    y_batch = torch.stack([data[i+1:i+context_length+1] for i in idxs]).to(device)
    return x_batch, y_batch


# calculate loss
@ torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# train
optimizer = torch.optim.Adam(model.parameters(), lr=learnint_rate)
tracked_losses = list()
for step in range (max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train', batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model state dictionary
torch.save(model.state_dict(), 'model-ckpt.pt')

# Generate
model.eval()
start = 'The salesperson'
start_ids = enc.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(enc.decode(y[0].tolist()))
print('---------------')


# # define embedding table
# input_embedding_lookup_table = nn.Embedding(max_token_value+1,d_model)  # (199853,64)

# # x_batch_embeddind
# x_batch_embedding = input_embedding_lookup_table(x_batch)
# # print(x_batch_embeddind.shape) # torch.Size([4, 16, 64])
# y_batch_embedding = input_embedding_lookup_table(y_batch)

# # positional encoding
# position_encoding_lookup_table = torch.zeros(context_length, d_model) # (16,64)
# pos = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)  # (16,1)
# # apply sin/cos
# div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (32,)
# position_encoding_lookup_table[:, 0::2] = torch.sin(pos * div_term) # (16,32)
# position_encoding_lookup_table[:, 1::2] = torch.cos(pos * div_term)  # (16,32)
# position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size, -1, -1) # (4,16,64)

# # add positional encoding to embedding
# x = x_batch_embedding + position_encoding_lookup_table
# y = y_batch_embedding + position_encoding_lookup_table

# # Query, Key, Value
# Wq = nn.Linear(d_model, d_model)
# Wk = nn.Linear(d_model, d_model)
# Wv = nn.Linear(d_model, d_model)
# Q = Wq(x) # (4,16,64)
# K = Wk(x) # (4,16,64)
# V = Wv(x) # (4,16,64)

# # apply multi-head attention
# num_heads = 4
# head_dim = d_model // num_heads
# Q = Q.reshape(batch_size, context_length, num_heads, head_dim).permute(0, 2, 1, 3) # (4,16,4,16)->(4,4,16,16)
# K = K.reshape(batch_size, context_length, num_heads, head_dim).permute(0, 2, 1, 3) 
# V = V.reshape(batch_size, context_length, num_heads, head_dim).permute(0, 2, 1, 3) 

# # scaled dot-product attention
# attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim) # (4,4,context_length,context_length)

# # apply mask
# # (1,1,context_length,context_length)
# mask = torch.tril(torch.ones(context_length, context_length)).unsqueeze(0).unsqueeze(0)
# mask = (1 - mask).bool()
# # print(mask)
# attention_scores = attention_scores.masked_fill(mask, float('-inf')) # 对true的部分进行-inf
# # print(attention_scores)

# # softmax
# attention_scores = F.softmax(attention_scores, dim=-1) # (4,4,context_length,context_length)
# # print(attention_scores)

# # apply attention_scores @ V
# A = attention_scores @ V
# # print(A.shape)

# # concatenate heads
# A = A.permute(0, 2, 1, 3).reshape(batch_size, context_length, d_model) # (4,16,64)
# # print(A.shape)
# mlp = nn.Linear(d_model, d_model)
# output = mlp(A) # (4,16,64)

# # residual connection and layer normalization
# output = output + x
# norm1 = nn.LayerNorm(d_model)
# output = norm1(output) # layer_norm不复用,有可学习的参数

# # apply feedforward network
# feedforward = nn.Sequential(nn.Linear(d_model, d_model * 4),
#                              nn.ReLU(),
#                              nn.Linear(d_model * 4, d_model))
# output = output + feedforward(output)
# norm2 = nn.LayerNorm(d_model)
# output = norm2(output)

# # apply final linear layer
# final_linear = nn.Linear(d_model, max_token_value+1)
# output = final_linear(output)
# # print(output.shape) # torch.Size([4, 16, 199854])

# logits = F.softmax(output, dim=-1)  # 16个词,每个词对应的199854个词的概率分布
# # predict_index = torch.argmax(logits, dim=-1)