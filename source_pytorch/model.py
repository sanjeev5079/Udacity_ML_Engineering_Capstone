# torch imports
import torch.nn.functional as F
import torch.nn as nn

class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network accepts the number of features as input, and produces
    a single sigmoid value that can be rounded to a label: 1 or 0, as output.
    
    Training Notes:
    Use BCELoss to train a binary classifier in PyTorch.
    BCELoss: binary cross entroy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss 
    """

    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of model.
        Input params required for loading code in train.py.
        :param input_features: the number of input features in training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs to produce
        """
        super(BinaryClassifier, self).__init__()

        # define all layers
        self.fc1 = nn.Linear(input_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(0.3)
        
        self.sig = nn.Sigmoid()


    def forward(self, x):
        """
        Perform a forward pass of the model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior of the network
        out = F.relu(self.fc1(x))
        out = self.drop(out)
        out = self.fc2(out)
        return self.sig(out)