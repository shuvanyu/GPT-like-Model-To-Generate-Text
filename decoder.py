import json

# Decoder: take a list of integers as input, and output a string
def decode(l):
    with open('vocab.json', 'r') as f:
        vocab_data = json.load(f)

    itos = vocab_data['itos']
    itos = {int(key): value for key, value in vocab_data['itos'].items()}
    
    return ''.join([itos[i] for i in l])