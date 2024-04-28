import json

# Encoder: take a string s as input, output list of integers
def encode(s):
    with open('vocab.json', 'r') as f:
        vocab_data = json.load(f)
    stoi = vocab_data['stoi']
    
    return [stoi[c] for c in s]