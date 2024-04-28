import torch
import torch.nn as nn
from torch.nn import functional as F
from vocabulary import *
from hyperparameters import *
from dataLoader import *


class Head(nn.Module):
    ''' One head of self-attention'''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        B, T, d = x.shape
        k = self.key(x)     # (B, T, d)
        q = self.query(x)   # (B, T, d)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * d**-0.5    # (B, T, d) @ (B, d, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))    # (B, T, T)
        wei = F.softmax(wei, dim = -1)             # (B, T ,T)
        
        # Apply dropout to prevent some of the tokens(nodes) 
        # to be dropped out in calculating the attention
        wei = self.dropout(wei) 

        # PErform weighted aggregation of values
        v = self.value(x)
        out = wei @ v    # (B, T, T) @ (B, T, d) -> (B, T, d)

        return out



class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Create a projection layer for the outputs of the attention layer
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        # The output from th attention layer passes through a projection layer
        # and then gets added to the reidual layer 
        out = self.proj(out)
        out = self.dropout(out)
        return out



#===============================================================================
class FeedForward(nn.Module):
    ''' a simple linear layer followed by Relu nonlinearity'''

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # fully connected linear layer for each token (independently).
            # All the tokens do this independently for themselves
            # from n_embed to 4*n_embed (from the paper)
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # This layer is the projection layer for the ffwd that adds to the residual pathway
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.net(x)



#===============================================================================
class Block(nn.Module):
    ''' Transformer Block: Communication (self-sttention) followed by computation
    This clubs the calculation of self attention as well as the feed forward'''

    def __init__(self, head_size, num_heads):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size//num_heads)
        self.ffwd = FeedForward(n_embed)
        # LayerNorm normalizes the features (i.e., the embedding vector of each token) to be
        # unit Gaussian at initialization
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # PreNorm notion
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



#===============================================================================
class LanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads the embedding vector corresponding to its index
        # from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # Position embedding table to store the positional embeddings for the position of
        # a character (token) in the sequence within the context-length
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # Create a sequence of blocks to replicate each steps
        self.blocks = nn.Sequential(*[Block(head_size, num_heads) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embed) # Final layer Norm

        # To go from token embeddings to logits, pass through final linear linear
        self.lm_head = nn.Linear(n_embed, vocab_size)


    # Forward Pass
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # input and target are both (B, T) tensor of integers
        # B = Batch Dimension (batch_size), T = Time dimension (block_size), d = Embedding dimension
        tok_embed = self.token_embedding_table(idx)   # (B, T, d) tensor => B (T*d) matrices packed together
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))   #(T,d)
        x = tok_embed + pos_embed   # (B, T, d)
        x = self.blocks(x)          # (B, T, d). Pass the input through the blocks
        x = self.ln_final(x)        # (B, T, d). Pass the input through the final layerNorm
        logits = self.lm_head(x)    # (B, T, vocab_size) tensor
        loss = None

        # If targets != None, meaning we are in the training phase
        # Else we are in the generation phase
        if targets != None:
            # Reshape the tensor in a 2-D array with shape(B*T, d).
            # wE do this to use the logits directly in the cross entropy loss in the nn module
            B, T, d = logits.shape
            logits = logits.view(B*T, d)
            # print(f'Here: {logits.shape}')
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss # shapes (B*T, d), float

    # Generate sequence(samples) from the NN
    def generate(self, idx, max_new_tokens):

        # input(idx) is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get the predictions upto the max_new_tokens time steps
            logits, _ = self(idx_cond) # Calls the foward method, (B, T, d)

            # Focus on the last time step in order to generate the logits for
            # the next token in the sequence, from the last token of the sequence
            # print(f'before:{logits.shape}')
            logits = logits[:, -1, :] # becomes (B, d)
            # print(f'after:{logits.shape}')

            # Apply softmax to get probabilities
            # Generate probabilities across all the tokens in the vocab (i.e., vocab_size)
            probs = F.softmax(logits, dim=-1)  # (B, d)
            #print(probs)

            # Sample an index randomly from the distribution
            next_idx = torch.multinomial(probs, num_samples=1) # (B, 1)
            # print(f'The randomly chosen next token: {next_idx}')

            # Append sampled index to the running sequence
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)
            # print(f'Generated Sequence: {idx}')

        return idx