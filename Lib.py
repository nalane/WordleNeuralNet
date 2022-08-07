import torch
from torch import nn

# Constants indicating knowledge of a letter
UNKNOWN = 0
INCORRECT = 1
MAYBE = 2
CORRECT = 3

# Class to allow for the hashing of an information vector
class Pair():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __hash__(self):
        return hash((str(self.x), str(self.y)))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Produce embeddings for each knowledge level
        self.embed = nn.Embedding(4, 64)

        # Merge knowledge level with letter position
        self.linear1 = nn.Linear(64, 64)

        # Convolve across the alphabet for each letter position
        self.nn1 = nn.Sequential(
            nn.Conv1d(64, 512, 26, stride=26),
            nn.ReLU(),
            nn.Conv1d(512, 256, 5),
            nn.Flatten(start_dim=-2)
        )

        # Convolve across the positions for each letter in the alphabet
        self.nn2 = nn.Sequential(
            nn.Conv1d(64, 512, 5, dilation=26),
            nn.ReLU(),
            nn.Conv1d(512, 256, 26),
            nn.Flatten(start_dim=-2)
        )

        # Fold information down into one prediction vector
        self.relu = nn.ReLU()
        self.linear2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 130)
        )

    # Push x through nn
    def forward(self, x):
        x = self.embed(x)
        x = self.linear1(x)
        x = torch.transpose(x, -1, -2)

        x1 = self.nn1(x)
        x2 = self.nn2(x)
        
        x = torch.cat((x1, x2), -1)
        x = self.relu(x)
        logits = self.linear2(x)
        
        new_shape = logits.shape[:-1] + (5, 26)
        logits = logits.reshape(new_shape)
        
        return logits
