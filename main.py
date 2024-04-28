import os
os.chdir('/content/drive/MyDrive/SeqGenerationwithTransformers')
import torch
import torch.nn as nn
from torch.nn import functional as F
from vocabulary import vocab
from hyperparameters import *
from dataLoader import *
from Transformer import LanguageModel
from train import *

torch.manual_seed(seed)

# Read file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab_size, data = vocab(text)

# Split the dataset
n = int(0.9*len(data))
train_data = data[:n] # 90% is train data
val_data = data[n:]   # 10% validation data

model = LanguageModel(vocab_size).to(device) # Instantiate the model object

train(model, train_data, val_data)
torch.save(model, 'model.pt')
