# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm1d, \
    Dropout, Conv1d, BatchNorm2d
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

        self.cnn_layers1 = Sequential(
            # Defining a 1D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2))
        self.cnn_layers2 = Sequential(
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2))
        self.dropout = Dropout()
        self.linear_layers1 = Linear(4*4*32, 1000)
        self.linear_layers2 = Linear(1000, 30)

    # Defining the forward pass
    def forward(self, x):
        # print("x0 _ inputs:", x.size())
        x = self.cnn_layers1(x)
        # print("x1:", x.size())
        x = self.cnn_layers2(x)
        # print("x2:", x.size())
        x = x.view(x.size(0), -1)
        # print("x3:", x.size())
        x = self.dropout(x)
        # print("x4:", x.size())
        x = self.linear_layers1(x)
        # print("x5:", x.size())
        x = self.linear_layers2(x)
        # print("x6:", x.size())
        return x
