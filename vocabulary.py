import torch
import json
import os
from encoder import encode

# Create a vocabulary of characters that occur in the text
def vocab(text):   
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) } # String-to-Integer
    itos = { i:ch for i,ch in enumerate(chars) } # Integer-to-String
    with open('vocab.json', 'w') as f:
        json.dump({'stoi': stoi, 'itos': itos}, f)

    # Represent the entire dataset in terms of integers / tokens and store it into a torch tensor
    data = torch.tensor(encode(text), dtype=torch.long)

    return vocab_size, data