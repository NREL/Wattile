import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(LSTM_Model, self).__init__()
        # Hidden Dimension
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building the LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Linear Layers
        self.fc1 = nn.Linear(hidden_dim, 100)
        self.fc2 = nn.Linear(100, 300)
        self.fc3 = nn.Linear(300, output_dim)

        self.device_id = device

    def forward(self, x):
        # Initializing the hidden state with zeros
        # (input, hx, batch_sizes)
        h0 = Variable(
            torch.zeros(
                self.layer_dim, x.size(0), self.hidden_dim, device=self.device_id
            )
        )

        c0 = Variable(
            torch.zeros(
                self.layer_dim, x.size(0), self.hidden_dim, device=self.device_id
            )
        )

        # One time step
        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = self.fc1(out[:, -1, :])
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)

        return out
