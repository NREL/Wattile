import torch.nn as nn


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.input_dim = input_dim
        # Linear function 1: 14 -> 56
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # initializing the weights with xavier weight initialization and bias as 0.01
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        # Non-linearity
        self.relu = nn.ReLU()

        # Linear function 2: 56 --> 56
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # initializing the weights with xavier weight initialization and bias as 0.01
        nn.init.xavier_uniform(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3: 56 --> 56
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # initializing the weights with xavier weight initialization and bias as 0.01
        nn.init.xavier_uniform(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)
        # Non-linearity 3
        self.relu3 = nn.ReLU()

        # Read-out layer (linear function): 56 --> 1
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        # initializing the weights with xavier weight initialization and bias as 0.01
        nn.init.xavier_uniform(self.fc4.weight)
        self.fc4.bias.data.fill_(0.01)

    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)

        return out
