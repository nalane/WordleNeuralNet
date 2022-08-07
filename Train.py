from Lib import *

import gc
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import f1_score

# Constants
BATCH_SIZE = 64
DATAFILE = "results.txt"

# Utility function for finding the number of trainable params in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# General testing function
def test(test_set, model, loss_fn):  
    # Evaluate 
    test_loss = 0
    f1 = 0
    model.eval()
    for X, y in test_set:
        with torch.no_grad():        
            # Make a predication
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            # Adjust loss and f1
            test_loss += loss_fn(pred, y.type(torch.float)).item()
            f1 += f1_score(pred.argmax(dim=2), y.argmax(dim=2), num_classes=26, mdmc_average='samplewise')
    
    return (test_loss, f1)

# General training function
def train(X, y, model, loss_fn, optimizer):
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y.type(torch.float))

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Train with validation
def validate_train(t_dataloader, v_dataloader, model, loss_fn, optimizer):
    size = len(t_dataloader.dataset)
    
    # Training phase
    model.train()
    for batch, (X, y) in enumerate(t_dataloader):            
        loss = train(X, y, model, loss_fn, optimizer)

        # Print progress
        if batch % 1000 == 0:
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
       
    # Validation phase
    size = len(v_dataloader)
    (loss, f1) = test(v_dataloader, model, loss_fn)
    
    # Print final vals
    loss /= size
    f1 /= size
    print(f"Test Error: \n F1 score: {f1:>0.3f}, Avg loss: {loss:>8f} \n")
    
    return (loss, f1)

# Load the datasets
class WordleDataset(Dataset):
    def __init__(self, filename, min_known):            
        # Load the file
        with open(filename, "rb") as file:
            self.ds = []
            
            for d in pickle.load(file):
                x = d.x
                known_count = len([n for n in x if n == CORRECT])
                if known_count >= min_known:
                    self.ds.append(d)
                
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        pair = self.ds[idx]
        x, y = pair.x, pair.y
        return torch.IntTensor(x), torch.IntTensor(y).reshape((5, 26))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train_at_known_letters(i, epochs, model_file, previous_file=None):
    # Create the model, perform transfer learning if old model available
    loss_fn = nn.CrossEntropyLoss()
    model = NeuralNetwork().to(device)
    if previous_file != None:
        model.load_state_dict(torch.load(previous_file))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        
    # Load the datasets
    training_data = WordleDataset("datasets/train.pkl", i)
    validation_data = WordleDataset("datasets/validation.pkl", i)

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE)
    
    # Train
    for t in range(epochs):
        print(f"Epoch: {t+1}; Known letters: {i}\n-------------------------------")
        (loss, f1) = validate_train(train_dataloader, validation_dataloader, model, loss_fn, optimizer)
    print("Done!")

    # Save learned params
    torch.save(model.state_dict(), model_file)
    
    # Make sure unneeded data is dropped
    del model
    del train_dataloader
    del validation_dataloader
    del training_data
    del validation_data
    gc.collect()

    return loss, f1

# Train models for different levels of input vector knowledge
model_file = ""
for i in range(5, -1, -1):
    # Perform training
    model_file = f"models/model_raw{i}.pth"
    loss, f1 = train_at_known_letters(i, 50, model_file)
    
    # Record results
    with open(DATAFILE, "a") as file:
        file.write(f"raw{i},{loss},{f1}\n")

# Perform transfer learning from no known letters to all known
previous_file = model_file
for i in range(1, 6):    
    # Perform training
    model_file = f"models/model_trans{i}.pth"
    loss, f1 = train_at_known_letters(i, 5, model_file, previous_file=previous_file)
    previous_file = model_file
    
    # Record results
    with open(DATAFILE, "a") as file:
        file.write(f"trans{i},{loss},{f1}\n")
