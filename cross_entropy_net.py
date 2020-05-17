import torch.nn as nn
import torch

class CrossEntropyNet(nn.Module):
    def __init__(self, n_states, hidden_size, n_outputs):
        super(CrossEntropyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_outputs)
        )
    
    def forward(self, x):
        return self.network(x)

