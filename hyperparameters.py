import torch

# Hyperparametrs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64  # how many independent sequences will bwe process in parallel
block_size = context_length = 256  # What is the max context length for predictions?
learning_rate = 3e-4
epochs = 8000
eval_interval = 500
n_embed = 256
num_heads = 8
head_size = n_embed
n_layer = 6
dropout_prob = 0.2
seed = 1337