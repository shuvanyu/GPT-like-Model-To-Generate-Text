from hyperparameters import *
torch.manual_seed(seed)

# Transform the dataset in batches to fully occupy the GPU
def get_batch(data):
    # Get random start index of a chunk (of size = block_length)
    # and store in a tensor of size batch_size
    # len(data)-block_size is the maximum index where a context window can start
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Iterate over all the start indices and stack the tensors
    # on top of each other
    x = torch.stack([data[i : i+block_size] for i in ix])

    # Get the targets for the corresponding input indices
    # Targets are shifted by one to the right
    y = torch.stack([data[i+1 : i+1+block_size] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y   # x, y = (batch_size, block_size)