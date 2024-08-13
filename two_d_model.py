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
    def __init__(self, input_size, output_size):
        super(FullyConnectedLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 80)
        self.fc4 = nn.Linear(80, output_size)  # Assuming you want an output of size 60

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.Tanh()(self.fc2(x))
        x=torch.nn.Tanh()(self.fc3(x))
        x=self.fc4(x)
        return x


class fc(torch.nn.Module):
    def __init__(self, input_shape, output_shape, num_layers, activation_last):
        super().__init__()
        self.activation_last=activation_last
        self.input_shape = input_shape
        self.output_shape = output_shape
        n = 100
        # self.activation = torch.nn.ReLU()
        # self.activation=torch.nn.LeakyReLU()
        self.activation = torch.nn.Tanh()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=self.input_shape, out_features=n, bias=True)])
        output_shape = n

        for j in range(num_layers):
            layer = torch.nn.Linear(
                in_features=output_shape, out_features=n, bias=True)
            # initializer(layer.weight)
            output_shape = n
            self.layers.append(layer)

        self.layers.append(torch.nn.Linear(
            in_features=output_shape, out_features=self.output_shape, bias=True))

    def forward(self, y):
        s=y
        for layer in self.layers:
            s = layer(self.activation(s))
        if self .activation_last:
            return self.activation(s)
        else:
            return s
        
class deeponet_f(nn.Module):
    def __init__(self, dim,f_shape, domain_shape,  p):
        super().__init__()

        n_layers = 4
        self.n = p

        self.attention1=SelfAttention2(input_dims=[1,1,1], hidden_dim=1)
        self.attention2=SelfAttention2(input_dims=[2,2,2], hidden_dim=1)
        
        self.branch1=FullyConnectedLayer(f_shape,f_shape)
        self.branch2=FullyConnectedLayer(2,1)
        self.trunk1=FullyConnectedLayer(2,f_shape)
        self.bias1 =fc( 3*f_shape, 1, n_layers, False)
       
    def forward(self, X):
        y,f,dom, mask=X

       
        branch2= self.branch2(self.attention2(dom,dom,dom, mask).squeeze(-1)).squeeze(-1)
        branch1= self.branch1(self.attention1(f.unsqueeze(-1),f.unsqueeze(-1),f.unsqueeze(-1), mask).squeeze(-1))

        # trunk = self.attention2(y.unsqueeze(-1),dom,y.unsqueeze(-1)).squeeze(-1)
        trunk=self.trunk1(y)
        bias = torch.squeeze(self.bias1(torch.cat((branch1,branch2,trunk),dim=1)))

        # return torch.sum(branch1*branch2*trunk, dim=-1, keepdim=False)+bias
        return torch.sum(branch1*branch2*trunk, dim=-1, keepdim=False)+bias




    def forward2(self, X):
        trunk,branch1,branch2, mask=X


        bias = torch.squeeze(self.bias1(torch.cat((branch1,branch2,trunk),dim=1)))

        # return torch.sum(branch1*branch2*trunk, dim=-1, keepdim=False)+bias
        return torch.sum(branch1*branch2*trunk, dim=-1, keepdim=False)+bias
    
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
            t1,t2, f1,f2, mask, v1, v2=X
   
            # return self.model1(X)
            return self.model1.forward2([t1,f1,v1,mask])+1J*self.model2.forward2([t2,f2,v2,mask])

