import time


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import numpy as np

from constants import Constants

from transformer import EncoderLayer, EncoderLayer2


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, output_size, activation=nn.Tanh()):
        super(FullyConnectedLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, output_size)
        self.activation=activation
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x=self.fc3(x)
        return x



class ConvNet(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(ConvNet, self).__init__()
        # Convolutional layers
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=3, stride=2)
        self.layer2 = nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3, stride=2)
        self.layer3 = nn.Conv2d(in_channels=60, out_channels=100, kernel_size=3, stride=2)

        self.activation=activation
        
        # Fully connected layers
        self.fc1 = nn.Linear(100, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 80)

    def forward(self, x):
        # Pass through convolutional layers with ReLU activation
     
        # nn.ReLU()
        x =self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        
        
        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers with ReLU activation
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        
        return x




class vanilla_deeponet_f(nn.Module):
    def __init__(self, dim,f_shape, domain_shape,  p):
        super().__init__()

        n_layers = 4
        self.n = p
        
        self.branch=ConvNet()

        self.trunk=FullyConnectedLayer(2,80)
       
    def forward(self, X):
        y,f,_, _,_=X
        y=y[:,:2]
        
        f=f.view(-1,1, 15,15)


        branch= self.branch(f)
        
        # trunk = self.attention2(y.unsqueeze(-1),dom,y.unsqueeze(-1)).squeeze(-1)
        trunk=self.trunk(y)
        return torch.sum(branch*trunk, dim=-1, keepdim=False)

class vannila_deeponet(nn.Module):  
        def __init__(self, dim,f_shape, domain_shape,  p):
            super().__init__()
            self.dim=dim
            self.p=p
            self.model1=vanilla_deeponet_f(dim,f_shape, domain_shape,  p)
            self.model2=vanilla_deeponet_f(dim,f_shape, domain_shape,  p)
         
        def forward(self, X):
   
            # return self.model1(X)
            return self.model1(X)+1J*self.model2(X)        

