import torch
from decoder import decode
from hyperparameters import device

model = torch.load('model.pt')
model.eval()

# Generate a sequence from the model
max_new_tokens = 5000
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens)[0].tolist()))