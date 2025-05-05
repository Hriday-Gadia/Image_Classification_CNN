import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T



# MLP class
class ModularMLP(nn.Module):
    def __init__(self, inputsize,activation,hiddenlayers,outputsize):
        super(ModularMLP, self).__init__()
        layers = []
        input_size = inputsize
        def get_activation(name):
            return {
                    "relu": nn.ReLU(),
                    "tanh": nn.Tanh(),
                    "sigmoid": nn.Sigmoid()
                    }.get(name.lower(), nn.ReLU())  # default to ReLU
        activation = get_activation(activation)
        for hidden_size in hiddenlayers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation)
            input_size = hidden_size

        layers.append(nn.Linear(input_size,outputsize))
        self.model = nn.Sequential(*layers)

    def forward(self ,x):
        x = x.view(x.size(0), -1)
        return self.model(x)

