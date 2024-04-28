from hyperparameters import *
from dataLoader import *

# Model evaluation mode
@torch.no_grad()
def evaluate(model, train_data, val_data):
    out = {}
    # Turn on the model evaluation mode which tells the model not to store the gradients
    model.eval()
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        total_loss = 0
        for _ in range(eval_interval):
            X, Y = get_batch(data)
            _, loss = model(X, Y)
            total_loss += loss.item()

        out[split] = total_loss / eval_interval

    model.train() # Return to the model training mode
    
    return out


def train(model, train_data, val_data):
    # Create a PyTorch Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for iter in range(epochs):
        if iter % eval_interval == 0:
            # Evaluate the model on the train and validation split
            losses = evaluate(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch(train_data)

        # Evaluate the loss
        logits, loss = model(xb, yb) # Calls the forward pass of the NN
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()