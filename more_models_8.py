
#  trained on l shapes as well
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
from utils import SelfAttention, SelfAttention2, track
from transformer import EncoderLayer, EncoderLayer2

class Snake(nn.Module):
    def __init__(self, a=1.0):
        super(Snake, self).__init__()
        self.a = a

    def forward(self, x):
        return torch.sin(x)
        # return x + (1.0 / self.a) * torch.sin(self.a * x) ** 2

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, output_size, activation=nn.Tanh()):
        super(FullyConnectedLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 80)
        self.fc4 = nn.Linear(80, output_size)  # Assuming you want an output of size 60
        self.activation=activation
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x=self.activation(self.fc3(x))
        x=self.fc4(x)
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
        self.fc1 = nn.Linear(2500, 80)
        self.fc2 = nn.Linear(80, 80)

    def forward(self, x):
        # Pass through convolutional layers with ReLU activation
     
        # nn.ReLU()
        x = torch.nn.ReLU()(self.layer1(x))
        x = torch.nn.ReLU()(self.layer2(x))
        x = torch.nn.ReLU()(self.layer3(x))
        
        
        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers with ReLU activation
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x

        
class deeponet_f(nn.Module):
    def __init__(self, dim,f_shape, domain_shape,  p):
        super().__init__()

        n_layers = 4
        self.n = p

        self.attention1=SelfAttention2(input_dims=[1,1,1], hidden_dim=1)
        self.attention2=SelfAttention2(input_dims=[2,2,2], hidden_dim=1)
        self.attention3=SelfAttention2(input_dims=[1,1,1], hidden_dim=1)
        self.attention4=SelfAttention2(input_dims=[1,1,1], hidden_dim=1)
        self.attention5=SelfAttention2(input_dims=[4,4,4], hidden_dim=1)
        self.attention6=SelfAttention2(input_dims=[2,2,2], hidden_dim=1)
        self.conv1=ConvNet()
    
        self.linear1=FullyConnectedLayer(225,225)
        self.linear2=FullyConnectedLayer(3,80, activation=Snake())
        self.linear3=FullyConnectedLayer(225,225)
        self.linear4=FullyConnectedLayer(225,80)
        self.linear5=FullyConnectedLayer(225,80, activation=Snake())
        self.linear6=FullyConnectedLayer(225,80)
        self.linear7=FullyConnectedLayer(225,80)
        
        
        
       

       
    def forward(self, X):
        y,f,dom, mask,sgnd=X
        f=f.view(-1,225,1)
        sgnd=sgnd.view(-1,225,1)
        dom=dom.view(-1,225,2)
        mask=mask.view(-1,225,225)
        x=torch.cat((f,dom,sgnd),dim=-1)
        branch=self.linear6(self.attention5(x,x,x,mask).squeeze(-1)).squeeze(-1)
        trunk=self.linear2(y).squeeze(-1)
        return torch.sum(branch*trunk, dim=-1, keepdim=False)

    def forward2(self, X):
        y,f,dom, mask,sgnd=X
        f=f.view(-1,225,1)
        dom=dom.view(-1,225,2)
        sgnd=sgnd.view(-1,225,1)
        mask=mask.view(-1,225,225)
        x=torch.cat((f,dom,sgnd),dim=-1)
        branch=self.linear6(self.attention5(x,x,x,mask).squeeze(-1)).squeeze(-1).repeat(y.shape[0],1)
        trunk=self.linear2(y).squeeze(-1)
        return torch.sum(branch*trunk, dim=-1, keepdim=False)
    
class deeponet(nn.Module):  
        def __init__(self, dim,f_shape, domain_shape,  p):
            super().__init__()
            self.dim=dim
            self.p=p
            self.model1=deeponet_f(dim,f_shape, domain_shape,  p)
            self.model2=deeponet_f(dim,f_shape, domain_shape,  p)
            
        def forward(self, X):
   
            # return self.model1(X)
            return self.model1(X)+1J*self.model2(X)

        def forward2(self, X):
   
            # return self.model1(X)
            return self.model1.forward2(X)+1J*self.model2.forward2(X)

        