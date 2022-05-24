import torch
import torch.nn as nn
from torch.autograd import Variable


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden Dimension
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building the RNN, specifying type of activation function
        # nonlinearity: relu or tanh
        #  batch_first: If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity="relu"
        )

        # Readout layer (Fully connected layer)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # self.hidden = Variable(torch.zeros(self.layer_dim, 960, self.hidden_dim))

    def forward(self, x):
        # Initializing the hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        # self.hidden = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        # h0 = h0.detach()
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).requires_grad_()
        # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim), requires_grad=True)
        # self.hidden = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        # self.hidden = self.hidden.detach()

        # One time step (the last one perhaps?)
        out, hn = self.rnn(x, h0)
        # out, hn = self.rnn(x, h0.detach())
        # out, _ = self.rnn(x)
        # out, self.hidden = self.rnn(x, self.hidden)
        # self.hidden = self.hidden.detach()
        # out, hn = self.rnn(x, self.hidden.detach_())

        # Indexing hidden state of the last time step
        # out.size() --> ??
        # out[:,-1,:] --> is it going to be 100,100
        out = self.fc(out[:, -1, :])
        # out.size() --> 100,1
        # print("Just did forward pass")
        return out
