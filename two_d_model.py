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
        self.fc3 = nn.Linear(100, output_size)
        self.activation=activation
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x=self.fc3(x)
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

class Vannila_FullyConnectedNet(nn.Module):
    def __init__(self,dim=2):
        super(Vannila_FullyConnectedNet, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(dim, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 80)

    def forward(self, x):
        # Pass through fully connected layers with Tanh activation
        activation=Snake()
        # tanh()
        x = activation(self.fc1(x))
        x = activation(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
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
        # start=time.time()
        branch1= self.branch1(self.attention1(f.unsqueeze(-1),f.unsqueeze(-1),f.unsqueeze(-1), mask).squeeze(-1))
        
        # trunk = self.attention2(y.unsqueeze(-1),dom,y.unsqueeze(-1)).squeeze(-1)
        trunk=self.trunk1(y)
        bias = torch.squeeze(self.bias1(torch.cat((branch1,branch2,trunk),dim=1)))
        # print(time.time()-start)
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

class deeponet_f_2(nn.Module):
    def __init__(self, dim,f_shape, domain_shape,  p):
        super().__init__()

        n_layers = 4
        self.n = p

        self.attention1=SelfAttention2(input_dims=[1,1,1], hidden_dim=1)

        
        self.branch1=FullyConnectedLayer(f_shape,f_shape)
        self.trunk1=FullyConnectedLayer(2,f_shape)
        self.bias1 =fc( 2*f_shape, 1, n_layers, False)
       
    def forward(self, X):
        y,f,dom, mask=X

       
     
        # start=time.time()
        branch1= self.branch1(self.attention1(f.unsqueeze(-1),f.unsqueeze(-1),f.unsqueeze(-1), mask).squeeze(-1))
        
        # trunk = self.attention2(y.unsqueeze(-1),dom,y.unsqueeze(-1)).squeeze(-1)
        trunk=self.trunk1(y)
        bias = torch.squeeze(self.bias1(torch.cat((branch1,trunk),dim=1)))
        # print(time.time()-start)
        # return torch.sum(branch1*branch2*trunk, dim=-1, keepdim=False)+bias
        return torch.sum(branch1*trunk, dim=-1, keepdim=False)+bias



    def forward2(self, X):
        trunk,branch1,branch2, mask=X


        bias = torch.squeeze(self.bias1(torch.cat((branch1,trunk),dim=1)))

        # return torch.sum(branch1*branch2*trunk, dim=-1, keepdim=False)+bias
        return torch.sum(branch1*trunk, dim=-1, keepdim=False)+bias
    
class deeponet2(nn.Module):  
        def __init__(self, dim,f_shape, domain_shape,  p):
            super().__init__()
            self.dim=dim
            self.p=p
            self.model1=deeponet_f_2(dim,f_shape, domain_shape,  p)
            self.model2=deeponet_f_2(dim,f_shape, domain_shape,  p)
         
        def forward(self, X):
   
            # return self.model1(X)
            return self.model1(X)+1J*self.model2(X)

        def forward2(self, X):
            t1,t2, f1,f2, mask, v1, v2=X
   
            # return self.model1(X)
            return self.model1.forward2([t1,f1,v1,mask])+1J*self.model2.forward2([t2,f2,v2,mask])


class vanilla_deeponet_f(nn.Module):
    def __init__(self, dim,f_shape, domain_shape,  p):
        super().__init__()

        n_layers = 4
        self.n = p
        
        self.branch=ConvNet()

        self.trunk=Vannila_FullyConnectedNet()
        self.bias =fc( 2*80, 1, n_layers, False)
       
    def forward(self, X):
        y,f,dom, mask=X
        f=f.view(-1,1, 15,15)


        branch= self.branch(f)
        
        # trunk = self.attention2(y.unsqueeze(-1),dom,y.unsqueeze(-1)).squeeze(-1)
        trunk=self.trunk(y)
        bias = torch.squeeze(self.bias(torch.cat((branch,trunk),dim=1)))
        # print(time.time()-start)
        # return torch.sum(branch1*branch2*trunk, dim=-1, keepdim=False)+bias
        return torch.sum(branch*trunk, dim=-1, keepdim=False)+bias

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
        

class conv_deeponet_f(nn.Module):
    def __init__(self, dim,f_shape, domain_shape,  p):
        super().__init__()

        n_layers = 4
        self.n = p
        # self.attention=SelfAttention2(input_dims=[2,2,2], hidden_dim=1)
        self.branch1=ConvNet(activation=Snake())
        self.branch2=ConvNet(activation=nn.Tanh())
        self.linear1=FullyConnectedLayer(160,80, activation=nn.Tanh())
        self.linear2=FullyConnectedLayer(160,80,activation=nn.Tanh())

        self.trunk=Vannila_FullyConnectedNet(dim=3)
        # self.bias =fc( 3*80, 1, n_layers, False)
       
    def forward(self, X):
        y,f,dom, mask=X
        f=f.view(-1,1, 15,15)
        dom=dom.unsqueeze(1)
        
        branch2= self.branch2(dom)
        branch1= self.branch1(f)
        trunk1=self.trunk(y)
        branch=self.linear1(torch.cat((branch1, branch2),dim=1))
        trunk=self.linear2(torch.cat((trunk1, branch2),dim=1))
        
        
        # trunk = self.attention2(y.unsqueeze(-1),dom,y.unsqueeze(-1)).squeeze(-1)
        
        # bias = torch.squeeze(self.bias(torch.cat((branch1, branch2,trunk),dim=1)))
        # print(time.time()-start)
        # return torch.sum(branch1*branch2*trunk, dim=-1, keepdim=False)+bias
        return torch.sum(branch*trunk, dim=-1, keepdim=False)

class conv_deeponet(nn.Module):  
        def __init__(self, dim,f_shape, domain_shape,  p):
            super().__init__()
            self.dim=dim
            self.p=p
            self.model1=conv_deeponet_f(dim,f_shape, domain_shape,  p)
            self.model2=conv_deeponet_f(dim,f_shape, domain_shape,  p)
         
        def forward(self, X):
   
            # return self.model1(X)
            return self.model1(X)+1J*self.model2(X)
        



