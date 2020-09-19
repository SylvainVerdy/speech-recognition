# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, Softmax, BatchNorm1d, Dropout, Conv1d
from torch.optim import Adam, SGD


class Net(Module):
    def __init__(self, batch_size, in_channels, h, w):
        """Initialize parameters and build model.
                Params
                ======
                    batch_size: size of the batch (number of samples loaded in the model)
                    in_channels:
                    h:
                    w:
                """
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 1D convolution layer
            Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
            ReLU(inplace=True),
            MaxPool1d(2),
            # Defining another 2D convolution layer
            Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool1d(2)

        )

        self.linear_layers = Sequential(
            Linear(64, 512),
            ReLU(inplace=True),
            Linear(512, 30),
            ReLU(inplace=True),
        )

    # Defining the forward pass
    def forward(self, x):
        print("x0 _ inputs:", x.size())
        x = self.cnn_layers(x)
        print("x1:", x.size())
        x = x.view(x.size(0), -1)
        print("x2:", x.size())
        x = self.linear_layers(x)
        return x