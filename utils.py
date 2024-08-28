import pandas as pd
from sklearn.metrics import pairwise_distances

from tqdm import tqdm
import datetime
from scipy.linalg import circulant

import math
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import gmres, lgmres
import random
from scipy.stats import gaussian_kde
import cmath
import os
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import memory_usage, profile
import scipy
from typing import List, Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn

from constants import Constants
from packages.my_packages import gmres2


class norms:
    def __init__(self): 
        pass
    @classmethod
    def relative_L2(cls,x,y):
        # 

        try:
            return (torch.linalg.norm(torch.real(x)-torch.real(y))/(torch.linalg.norm(torch.real(y))+1e-10)
         +torch.linalg.norm(torch.imag(x)-torch.imag(y))/(torch.linalg.norm(torch.imag(y))+1e-10)       
        )
        except:
            return torch.linalg.norm(x-y)/(torch.linalg.norm(y)+1e-10)
    @classmethod
    def relative_L1(cls,x,y):
        return torch.nn.L1Loss()(x,y)/(torch.nn.L1Loss(y,y*0)+1e-10)
    
def grf(l_x, n, seed=0, mu=0, sigma=0.1):
    np.random.seed(seed)
    A=np.array([np.random.normal(mu, sigma,n) for i in range(l_x) ]).T

    # [plt.plot(domain, np.sqrt(2)*A[i,:]) for i in range(n)]
    # plt.show(block=False)
    # torch.save(A, Constants.outputs_path+'grf.pt')
    return A

def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))



def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def polygon_centre_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_centroid(vertices):
    A = polygon_centre_area(vertices)
    x = vertices[:, 0]
    y = vertices[:, 1]
    Cx = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / 6 / A
    Cy = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / 6 / A
    return Cx, Cy


def map_right(p1, p2, p3):
    B = np.array([[p1[0]], [p1[1]]])
    A = np.array([[p2[0] - B[0], p3[0] - B[0]], [p2[1] - B[1], p3[1] - B[1]]])

    return np.squeeze(A), B


def is_between(p1, p2, point):
    crossproduct = (point[1] - p1[1]) * (p2[0] - p1[0]) - (point[0] - p1[0]) * (
        p2[1] - p1[1]
    )

    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > 1e-10:
        return False

    dotproduct = (point[0] - p1[0]) * (p2[0] - p1[0]) + (point[1] - p1[1]) * (
        p2[1] - p1[1]
    )
    if dotproduct < 0:
        return False

    squaredlengthba = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (
        p2[1] - p1[1]
    )
    if dotproduct > squaredlengthba:
        return False

    return True


def on_boundary(point, geo):
    for i in range(len(geo.__dict__["paths"])):
        p1 = geo.__dict__["paths"][i].__dict__["x0"]
        p2 = geo.__dict__["paths"][i].__dict__["x1"]
        if is_between(p1, p2, point):
            return True
    return False





def spread_points(subset_num,X):
    
    x=X[:,0]
    y=X[:,1]
    total_num = x.shape[0]
    xy = np.vstack([x, y])
    dens = gaussian_kde(xy)(xy)

    # Try playing around with this weight. Compare 1/dens,  1-dens, and (1-dens)**2
    weight = 1 / dens
    weight /= weight.sum()

    # Draw a sample using np.random.choice with the specified probabilities.
    # We'll need to view things as an object array because np.random.choice
    # expects a 1D array.
    dat = xy.T.ravel().view([('x', float), ('y', float)])
    # subset = np.random.choice(dat, subset_num, p=weight)
    subset = np.random.choice(dat, subset_num)
    return np.vstack((subset['x'], subset['y'])).T
    



def np_to_torch(x):
    return torch.tensor(x, dtype=Constants.dtype)


def save_file(f, dir, name):

    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
    torch.save(f, dir + name + ".pt")
    return dir + name + ".pt"


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, log_path, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.path = log_path

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                self.path+'best_model.pth',
            )


def save_plots(train_loss, valid_loss, test_loss, metric_type: str, dir_path):

    # accuracy plots
    fig, ax = plt.subplots(1, 2)
    # plt.figure(figsize=(10, 7))
    ax[0].plot(train_loss[1:], color="orange", linestyle="-", label="train")
    ax[0].plot(valid_loss[1:], color="red", linestyle="-", label="validataion")
    ax[0].set(xlabel='Epochs', ylabel=metric_type)
    ax[0].legend(loc="upper right")

    ax[1].plot(test_loss, color="blue", linestyle="-", label="test")
    ax[1].set(xlabel='Epochs', ylabel=metric_type)
    ax[1].legend(loc="upper right")

    fig.suptitle("metric type: "+metric_type)
    isExist = os.path.exists(dir_path+'figures')
    if not isExist:
        os.makedirs(dir_path+'figures')

    plt.savefig(dir_path + "figures/" + metric_type+".png")
    # plt.show(block=False)







def calc_min_angle(geo):
    seg1 = []
    for i in range(len(geo.__dict__["paths"])):
        p1 = geo.__dict__["paths"][i].__dict__["x0"]
        p2 = geo.__dict__["paths"][i].__dict__["x1"]
        seg1.append(p1)

    angle = []
    for i in range(len(seg1)):
        p1 = seg1[i % len(seg1)]
        p2 = seg1[(i - 1) % len(seg1)]
        p3 = seg1[(i + 1) % len(seg1)]
        angle.append(
            np.dot(p2 - p1, p3 - p1)
            / (np.linalg.norm(p2 - p1) * np.linalg.norm(p3 - p1))
        )
 
    return np.arccos(angle)




def solve_helmholtz(M, interior_indices, f):
    A = -M[interior_indices][:, interior_indices] - Constants.k * scipy.sparse.identity(
        len(interior_indices)
    )
    #    x,y,e=Gauss_zeidel(A,f[interior_indices])
    #    print(e)
    return scipy.sparse.linalg.spsolve(A, f[interior_indices])


# solve_helmholtz(M, interior_indices, f)


def extract_path_from_dir(dir):
    raw_names = next(os.walk(dir), (None, None, []))[2]
    return [dir + n for n in raw_names if n.endswith(".pt")]








def complex_version(v):
    assert v.size == 2
    r = np.sqrt(v[0] ** 2 + v[1] ** 2)
    theta = np.arctan2(v[1], v[0])
    return r*cmath.exp(1j*theta)




def save_figure(X, Y, titles, names, colors):

    # accuracy plots
    fig, ax = plt.subplots(1, len(X))
    for j in range(len(X)):
        ax[j].scatter(X[j],Y[j])

    plt.savefig(Constants.fig_path + "figures/" + ".eps",format='eps',bbox_inches='tight')
    plt.show(block=False)

 



def step_fourier(L,Theta):
    # fourier expansion of simple function in [0,1]
    # L is segments lengths
    # Theta is the angle function's values on the segments
    N=50
    x=[0]+[np.sum(L[:k+1]) for k in range(len(L))]
    a0=np.sum([l*theta for l,theta in zip(L,Theta)])
    a1=[2*np.sum([L[i]*Theta[i]*(np.sin(2*math.pi*n*x[i+1])-np.sin(2*math.pi*n*x[i]))/(2*math.pi*n) 
                  for i in range(len(L))]) for n in range(1,N)]
    a2=[2*np.sum([L[i]*Theta[i]*(-np.cos(2*math.pi*n*x[i+1])+np.cos(2*math.pi*n*x[i]))/(2*math.pi*n)
                   for i in range(len(L))]) for n in range(1,N)]
    coeff=[a0]
    for i in range(N-1):
        coeff.append(a1[i])
        coeff.append(a2[i])

    return np.array(coeff)

def save_uniqe(file, path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        
    uniq_filename = (
            str(datetime.datetime.now().date())
            + "_"
            + str(datetime.datetime.now().time()).replace(":", ".")
        )
    torch.save(file, path+uniq_filename+'.pt') 

def save_eps(name):
        plt.savefig(Constants.tex_fig_path+name, format='eps',bbox_inches='tight')
        plt.show(block=False)


def plot_figures(ax,y, **kwargs):
    d=kwargs
    try:
        ax.plot(y, color=d['color'],  label=d['label'])
        ax.legend()
    except:
        ax.plot(y, color=d['color'])    
    try:
        ax.set_title(d['title'])
    except:
        pass    
    ax.set_xlabel(d['xlabel'])
    ax.set_ylabel(d['ylabel']) 
    try:
        ax.text(320, d['text_hight'], f'err={y[-1]:.2e}', c=d['color'])
    except:
        pass    
    
      
       
def closest(set1,p):
    temp=np.argmin(np.array([np.linalg.norm(x-p) for x in set1]))
    return set1[temp], temp


class rect_solver:
    def __init__(self,x,y,l,where,robin=None):
        self.x=x
        self.robin=robin
        self.dx=x[1]-x[0]
        self.dy=y[1]-y[0]
        self.y=y
        self.l=l
        self.where=where
        if self.where:
            self.X, self.Y = np.meshgrid(x[1:], y[1:-1], indexing='ij')
        else:
            self.X, self.Y = np.meshgrid(x[:-1], y[1:-1], indexing='ij')
            
    def  calc_D_x(self):   
        Nx = len(self.x[:-1])

        
        kernel = np.zeros((Nx, 1))
        kernel[-1] = 1.
        kernel[0] = -2.
        kernel[1] = 1.
        D2 = circulant(kernel).astype(complex)
        D2[0, -1] = 0.
        D2[-1, 0] = 0.
        if self.where:
            D2[-1,-1]=-2-2*self.dx*self.l
            D2[-1,-2]=2
        else:    
            D2[0,0]=-2-2*self.dx*self.l
            D2[0,1]=2

        return D2/self.dx/self.dx
    
    def  calc_D_y(self):   

        Ny= len(self.y[1:-1])
        
        kernel = np.zeros((Ny, 1))
        kernel[-1] = 1.
        kernel[0] = -2.
        kernel[1] = 1.
        D2 = circulant(kernel).astype(complex)
        D2[0, -1] = 0.
        D2[-1, 0] = 0.
        return D2/self.dy/self.dy
    
    def calc_D(self):
        return csr_matrix(kron(self.calc_D_x(), identity(len(self.y)-2)),dtype=np.cfloat)+csr_matrix(kron(identity(len(self.x)-1), self.calc_D_y()), dtype=np.cfloat)

    
    def calc_bc(self):
        BC=(self.X*0).astype(complex)
        for i in range(len(self.x[1:])):
            for j in range(len(self.y[1:-1])):
                if self.where:
                    if abs(self.X[i,j]-self.x[-1])<1e-12:
                            BC[i,j]=2*self.robin[j]/(self.dx)
                            
                else:   
                    if abs(self.X[i,j]-self.x[0])<1e-12:
                        BC[i,j]=-2*self.robin[j]/(self.dx) 
        
        return BC.flatten()       
            
def calc_Robin(u,dx,l,side):
    if side:
        # return (u[-1,:]-u[-2,:])/dx-l*u[-1,:]
        return (3*u[-1,:]-4*u[-2,:]+u[-3,:])/(2*dx)-l*u[-1,:]
    else:
        return (-3*u[0,:]+4*u[1,:]-u[2,:])/(2*dx)+l*u[0,:]
        # return (u[1,:]-u[0,:])/dx+l*u[0,:]

def solve_subdomain(x,y,F,bc,l,side):
    # X, Y = np.meshgrid(x[1:], y[1:-1], indexing='ij') 
    # bc=np.cos(0.5)*np.sin(math.pi*y[1:-1])
    # u=np.sin(X)*np.sin(math.pi*Y)
    solver=rect_solver(x,y,l,side,bc)
    M=solver.calc_D()
    term=Constants.k* scipy.sparse.identity(M.shape[0])
    G=solver.calc_bc()
    # print((M@(u.flatten())+G-(-u-math.pi**2*u).flatten()).reshape((len(x)-1,len(y)-2)))
    return scipy.sparse.linalg.spsolve(M+term, -G+F)

def solve_subdomain2(x,y,F,bc,l,side):
    # X, Y = np.meshgrid(x[1:], y[1:-1], indexing='ij') 
    # bc=np.cos(0.5)*np.sin(math.pi*y[1:-1])
    # u=np.sin(X)*np.sin(math.pi*Y)
    solver=rect_solver(x,y,l,side,bc)
    M=solver.calc_D()
    term=Constants.k* scipy.sparse.identity(M.shape[0])
    G=solver.calc_bc()
    # print((M@(u.flatten())+G-(-u-math.pi**2*u).flatten()).reshape((len(x)-1,len(y)-2)))
    return M+term, -G+F
 
def subsample(x=None,y=None):
    n=11
    x0=np.linspace(0,1,n)
    y0=np.linspace(0,1,n)
    m=81
    x=np.linspace(0,1,m)
    y=np.linspace(0,1,m)
    k=(int((m-1)/(n-1)))
    print(x[0::k]-x0)
   
  
def upsample(f,n):
    f=torch.tensor(f)
    f=torch.reshape(f,(1,1,n,n))
    return (torch.squeeze(nn.Upsample(scale_factor=2, mode='bilinear')(f)).numpy()).flatten()
    
    
def generate_random_matrix(rows, seed):
    # Generate random values from a standard normal distribution
    assert seed>-1
    np.random.seed(seed)
        
    random_values = np.random.randn(rows)
    
    # Calculate the mean and standard deviation of the generated values
    mean = np.mean(random_values)
    std_dev = np.std(random_values)
    
    # Adjust the values to have a mean of 0 and a variance of 1
    normalized_values = (random_values - mean) / std_dev
    
    return normalized_values

def bilinear_upsample(f):
    n = len(f)
    upsampled_f = np.zeros(2*n)
    
    # Upsample every other pixel
    upsampled_f[::2] = f
    
    # Perform linear interpolation for every missing pixel
    for i in range(1, n):
        upsampled_f[2*i] = (f[i-1] + f[i]) / 2.0
    
    return upsampled_f    






# Example usage:

class SelfAttention(nn.Module):
    def __init__(self, input_dims, hidden_dim, seed):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        torch.manual_seed(seed)
        self.query = nn.Linear(input_dims[0], hidden_dim, bias=False)
        self.key = nn.Linear(input_dims[1], hidden_dim, bias=False)
        self.value = nn.Linear(input_dims[2], hidden_dim, bias=False)
    

       
        
    def forward(self, x1,x2,x3, mask=None):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x3)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        output = torch.matmul(attention_weights, v)
        return output
# x=torch.tensor(np.array([1,2]), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
# y=torch.tensor(np.array([2,1]), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
# d1=torch.tensor(np.array([-1,3]), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
# d2=torch.tensor(np.array([3,-1]), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
# y0=torch.tensor(np.array([3,-1,2]), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
# l1=SelfAttention([1,1,1],1,seed=1) 
# l2=SelfAttention([1,1,1],1,seed=2) 
# print(l1(x,x,d1))
# print(l1(y,y,d2))
  
# print(l(dom,dom,dom).shape )
def custom_softmax(x, dim):
    # Compute the exponentials of the input tensor
    exp_x = torch.exp(x)
    exp_x[torch.isinf(exp_x)]=1e30
    
    # Sum the exponentials along the specified dimension
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    
    # Divide the exponentials by the sum of exponentials
    softmax_x = exp_x / (sum_exp_x+1e-14)
    
    return softmax_x
class SelfAttention2(nn.Module):
    def __init__(self, input_dims, hidden_dim, include_weigts=True):
        super(SelfAttention2, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(input_dims[0], hidden_dim,bias=False)
        self.key = nn.Linear(input_dims[1], hidden_dim,bias=False)
        self.value = nn.Linear(input_dims[2], hidden_dim,bias=False)
        self.include_weights=include_weigts
        
    def forward(self, x1,x2,x3, mask):
        if self.include_weights:
            q = self.query(x1)
            k = self.key(x2)
            v = self.value(x3)
        else:
            q=x1
            k=x2
            v=x3

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
        
        # attention_weights = F.softmax(attention_scores+mask, dim=-1)
        attention_weights=custom_softmax(attention_scores+mask, dim=-1)
        
        output = torch.matmul(attention_weights, v)
        return output
    


# dom= torch.randn(2, 225,2 )  # Batch size 4, sequence length 10, input dimension 64
# dom=torch.randn(2, 225,2 )
# attention1=SelfAttention1(input_dims=[2,2,2], hidden_dim=1)
# print(attention1(dom,dom,dom,dom).shape)
# attention2=SelfAttention2(input_dims=[2,2,2], hidden_dim=1)
# attention1(dom,dom,dom)
# x2 = torch.randn(2, 4, 1)   # Batch size 4, sequence length 8, input dimension 64
# x3 = torch.randn(2, 4,1 )  # Batch size 4, sequence length 12, input dimension 64

# # # # # # # Apply self-attention to each input
# attention_layer = SelfAttention(input_dim=4, hidden_dim=1)
# print(attention_layer(x1,x2,x3,[[1],[2]]).shape  )


def plot_cloud(X,Y,color):
    from scipy.spatial import ConvexHull
    points = np.column_stack((X, Y))
    hull = ConvexHull(points)
    convexlist = hull.simplices
    for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=color)
    
def block_matrix(A,B):
    n_rows = A.shape[0] + B.shape[0]
    n_cols = A.shape[1] + B.shape[1]

# Create zero matrices for the off-diagonal blocks
    zero_matrix_top = np.zeros((A.shape[0], B.shape[1]))
    zero_matrix_bottom = np.zeros((B.shape[0], A.shape[1]))

# Construct the block matrix
    block_matrix = np.block([[A, zero_matrix_top], [zero_matrix_bottom, B]]) 
    return block_matrix   

from packages.my_packages import  interpolation_2D
def subsample(f,X,Y):
    f=interpolation_2D(X,Y,f)
    return f
    
def evaluate_model(f,valid_indices,d,d_super,NN,NN2,X,Y, dom,mask):
            f_real=(f).real
            f_imag=(f).imag
            
            func_real=interpolation_2D(d_super.X,d_super.Y,f_real)
            func_imag=interpolation_2D(d_super.X,d_super.Y,f_imag)
            f_real=np.array(func_real(d.X,d.Y))
            f_imag=np.array(func_imag(d.X,d.Y))
            
            mu_real=np.mean(f_real)
            s_real=np.std(f_real)
            mu_imag=np.mean(f_imag)
            s_imag=np.std(f_imag)
            f_ref_real=np.zeros(Constants.n**2)
            f_ref_imag=np.zeros(Constants.n**2)
            
            
            f_ref_real[valid_indices]=(f_real-mu_real)/s_real
            f_ref_imag[valid_indices]=(f_imag-mu_imag)/(s_imag+1e-14)
            corr_real=(NN(f_ref_real,X,Y, dom, mask)+mu_real*NN2(f_ref_real*0+1,X,Y,dom,mask)/s_real)*s_real
            corr_imag=(NN(f_ref_imag,X,Y, dom, mask)+mu_imag*NN2(f_ref_real*0+1,X,Y,dom,mask)/(s_imag+1e-14))*s_imag
            # corr_real=(NN(f_ref_real,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_real)/s_real)*s_real
            # corr_imag=(NN(f_ref_imag,X,Y, dom, mask)+scipy.sparse.linalg.spsolve(A, b*0+mu_imag)/s_imag)*s_imag
            corr=corr_real+1J*corr_imag
            return corr

# A = pyamg.gallery.poisson((30,30), format='csr')/((1/31)**2)  # 2D Poisson problem on 500x500 grid
# A=A-201*scipy.sparse.identity(A.shape[0])
# f=np.random.rand(A.shape[0])  
# x, exitCode = scipy.sparse.linalg.gmres(A, f, tol=1e-13, maxiter=800)
# print(np.linalg.norm(A@x-f)/np.linalg.norm(f))
# print(A.shape)
# l,v=scipy.sparse.linalg.eigs(A, k=2,which='SR')
# print(l)
# ml = pyamg.smoothed_aggregation_solver(A)                   # construct the multigrid hierarchy
#                           # print hierarchy information
# b = np.random.rand(A.shape[0])                      # pick a random right hand side
# x = ml.solve(b, tol=1e-10, maxiter=1000)                          # solve Ax=b to a tolerance of 1e-10
# print("residual: ", np.linalg.norm(b-A*x)) 

# n=1000
# m=1
# x=np.linspace(0,1,n)
# cov=np.zeros((n,n))
# mean=np.zeros(n)
# for i in range(n):
#     for j in range(n):
#         cov[i,j]=np.exp(-(x[i]-x[j])**2)

# f=np.random.multivariate_normal(mean, cov,size=m)
# print(np.mean(f))

def generate_grf(X,Y,n_samples,sigma=0.1,l=0.1,mean=0,seed=1):

    n=len(X)
    mean=np.zeros(n)+mean
    cov=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            s1=np.array([X[i],Y[i]])
            s2=np.array([X[j],Y[j]])
            cov[i,j]=sigma*np.exp(-(np.linalg.norm(s1-s2))**2/(2*l**2))
    np.random.seed(seed)
    return np.random.multivariate_normal(mean, cov, n_samples)



def fft(data,nx,ny):
    x=np.arange(nx)
    y=np.arange(ny)
    X,Y= np.meshgrid(x,y)

    data_wo_DC= data- np.mean(data)

    spectrum = np.fft.fftshift(np.fft.fft2(data)) 
    spectrum_wo_DC = np.fft.fftshift(np.fft.fft2(data_wo_DC)) 

    freqx=np.fft.fftshift(np.fft.fftfreq(nx,1))   #q(n, d=1.0)
    freqy=np.fft.fftshift(np.fft.fftfreq(ny,1))   
 
    return spectrum, freqx, freqy

import time
import os
import psutil


def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def track(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()/1e6
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()/1e6
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before,
            elapsed_time))
        return result
    return wrapper

class IterationCounter:
    def __init__(self):
        self.num_iterations = 0

    def __call__(self, rk=None):
        self.num_iterations += 1
def solve_gmres(A,b, x0, maxiter=100000, tol=1e-10):
    start=time.time()
    iteration_counter = IterationCounter()
    # x_s=x0.copy()
    # x1,_=lgmres(A, b,x0=x_s,tol=1e-13, maxiter=2, callback=iteration_counter)
    # err1=np.linalg.norm(A@x1-b)/np.linalg.norm(b)
    
    # x2=x_s
    # for i in range(2):
    #     x2,_=lgmres(A, b,x0=x2,tol=1e-13, maxiter=1, callback=iteration_counter)
    # err2=np.linalg.norm(A@x2-b)/np.linalg.norm(b)
    def call():
        x, exit_code = gmres(A, b,x0=x0,tol=tol, maxiter=maxiter, callback=iteration_counter)
        err=np.linalg.norm(A@x-b)/np.linalg.norm(b)
        return x,err
   
    x,err=call()
    time_counter=time.time()-start
    

    # Output the solution and the number of iterations
    return x, err, iteration_counter.num_iterations, time_counter


def find_sub_indices(X,Y,X_ref,Y_ref):
    valid_indices=[]
    original_points=[(X_ref[i],Y_ref[i]) for i in range(len(X_ref))]
    points=np.array([(X[i],Y[i]) for i in range(len(X))])
    for j,p in enumerate(points):
            dist=[np.linalg.norm(np.array(p)-point) for point in original_points]
            if np.min(dist)<1e-14:
                valid_indices.append(j)

    return valid_indices


import psutil
import time


# from scipy.io import mmread
# A = mmread('helm3d01.mtx')
# b=np.random.rand(A.shape[0])

# x, exit_code =lgmres(A, b,x0=b*0,tol=1e-13, maxiter=1000)
# err=np.linalg.norm(A@x-b)/np.linalg.norm(b)
# print(err)
 
# print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)  
# A = np.random.rand(10000,10000)
# print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)    
# del A
# print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)    
   
   

# A = np.random.rand(10,10)
# b = np.random.rand(10)
# call(A,b)    


#     x, exit_code = gmres(A, b,tol=1e-10, maxiter=2000)
#     print(exit_code)
#     return 
# if __name__=='__main__':    
#     A=np.random.rand(1000,1000)
#     b=np.random.rand(1000)
#     call(A,b)    

# from scipy.sparse import csr_array, tril
# A = csr_array([[1, 2, 0, 0], [4, 5, 0, 6], [0, 0, 8, 9], [1,2,3,4]],
#               dtype='int32')
# L=tril(A,k=0)
# U=A-L
# print(A.todense())
# print(L.todense())



import torch
import torch.nn as nn
from torchvision import models, transforms
from concurrent.futures import ThreadPoolExecutor

# Function to perform forward pass on the data using the model
def forward_pass(model, input_tensor, device):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
    
    return outputs


def create_D2(x):
    Nx = len(x[1:-1])
    dx = x[1] - x[0]
    kernel = np.zeros((Nx, 1))
    kernel[-1] = 1
    kernel[0] = -2
    kernel[1] = 1
    D2 = circulant(kernel)
    D2[0, -1] = 0
    D2[-1, 0] = 0

    return D2/dx/dx


def create_second(x):
    Nx = len(x)
    dx = x[1] - x[0]
    kernel = np.zeros((Nx, 1))
    kernel[-1] = 1
    kernel[0] = -2
    kernel[1] = 1
    D2 = circulant(kernel)
    D2[0, -1] = 0
    D2[-1, 0] = 0

    return D2/dx/dx


def create_Ds2(x, y):
    return csr_matrix(kron(create_D2(x), identity(len(y)-2)))+ csr_matrix(kron(identity(len(x)-2), create_D2(y)))
def create_Helm(x,y):
    D=create_Ds2(x,y)
    return D+Constants.k*scipy.sparse.identity(D.shape[0])

def GS_method(A, Y, X):
    n = A.shape[0] # to find no of rows or col of sq matrix A
    for j in range(0, n):
        #old_val = X[i]
        summ_val = Y[j]
        for i in range(0, n):
            if (j!=i):
                summ_val-=A[j,i] * X[i]
                #print(summ_val)
        X[j] = summ_val/A[j,j]
        #print(X[i]) 
    return X

def np_wn(pnts, poly, return_winding=False):
    if np.sum([is_between(poly[i],poly[i+1],pnts) for i in range(len(poly)-1)])>0:
            return 1
    """Return points in polygon using a winding number algorithm in numpy.

    Parameters
    ----------
    pnts : Nx2 array
        Points represented as an x,y array.
    poly : Nx2 array
        Polygon consisting of at least 4 points oriented in a clockwise manner.
    return_winding : boolean
        True, returns the winding number pattern for testing purposes.  Keep as
        False to avoid downstream errors.

    Returns
    -------
    The points within or on the boundary of the geometry.

    References
    ----------
    `<https://github.com/congma/polygon-inclusion/blob/master/
    polygon_inclusion.py>`_.  inspiration for this numpy version
    """
    x0, y0 = poly[:-1].T  # polygon `from` coordinates
    x1, y1 = poly[1:].T   # polygon `to` coordinates
    x, y = pnts.T         # point coordinates
    y_y0 = y[:, None] - y0
    x_x0 = x[:, None] - x0
    diff_ = (x1 - x0) * y_y0 - (y1 - y0) * x_x0  # diff => einsum in original
    chk1 = (y_y0 >= 0.0)
    chk2 = np.less(y[:, None], y1)  # pnts[:, 1][:, None], poly[1:, 1])
    chk3 = np.sign(diff_).astype(int)
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    wn = pos - neg
    out_ = np.nonzero(wn)
    if return_winding:
        return out_, wn
    return wn

#   a Point is represented as a tuple: (x,y)

#===================================================================

# is_left(): tests if a point is Left|On|Right of an infinite line.

#   Input: three points P0, P1, and P2
#   Return: >0 for P2 left of the line through P0 and P1
#           =0 for P2 on the line
#           <0 for P2 right of the line
#   See: the January 2001 Algorithm "Area of 2D and 3D Triangles and Polygons"

def is_left(P0, P1, P2):
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

#===================================================================

# cn_PnPoly(): crossing number test for a point in a polygon
#     Input:  P = a point,
#             V[] = vertex points of a polygon
#     Return: 0 = outside, 1 = inside
# This code is patterned after [Franklin, 2000]

def cn_PnPoly(P, V):
    cn = 0    # the crossing number counter

    # repeat the first vertex at end
    V = tuple(V[:])+(V[0],)

    # loop through all edges of the polygon
    for i in range(len(V)-1):   # edge from V[i] to V[i+1]
        if ((V[i][1] <= P[1] and V[i+1][1] > P[1])   # an upward crossing
            or (V[i][1] > P[1] and V[i+1][1] <= P[1])):  # a downward crossing
            # compute the actual edge-ray intersect x-coordinate
            vt = (P[1] - V[i][1]) / float(V[i+1][1] - V[i][1])
            if P[0] < V[i][0] + vt * (V[i+1][0] - V[i][0]): # P[0] < intersect
                cn += 1  # a valid crossing of y=P[1] right of P[0]

    return cn % 2   # 0 if even (out), and 1 if odd (in)

#===================================================================

# wn_PnPoly(): winding number test for a point in a polygon
#     Input:  P = a point,
#             V[] = vertex points of a polygon
#     Return: wn = the winding number (=0 only if P is outside V[])

def wn_PnPoly(P, V):
    if np.sum([is_between(V[i],V[i+1],P) for i in range(len(V)-1)])>0:
        return 1
    wn = 0   # the winding number counter

    # repeat the first vertex at end
    V = tuple(V[:]) + (V[0],)

    # loop through all edges of the polygon
    for i in range(len(V)-1):     # edge from V[i] to V[i+1]
        if V[i][1] <= P[1]:        # start y <= P[1]
            if V[i+1][1] > P[1]:     # an upward crossing
                if is_left(V[i], V[i+1], P) > 0: # P left of edge
                    wn += 1           # have a valid up intersect
        else:                      # start y > P[1] (no test needed)
            if V[i+1][1] <= P[1]:    # a downward crossing
                if is_left(V[i], V[i+1], P) < 0: # P right of edge
                    wn -= 1           # have a valid down intersect
    return wn

def generate_polygon_vertices(n):
    if n < 3:
        raise ValueError("A polygon must have at least 3 edges.")

    vertices = []
    angle_increment = 2 * math.pi / n

    for i in range(n):
        angle = i * angle_increment
        x = math.cos(angle)
        y = math.sin(angle)
        vertices.append((6/14*x+0.5, 6/14*y+0.5))
    
    vertices.append(vertices[0])
    return np.array(vertices)

# V=generate_polygon_vertices(40)
# plt.scatter(V[:,0],V[:,1])
# plt.show()

def distance(a,b,p):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dr2 = float(dx ** 2 + dy ** 2)

    lerp = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / dr2
    if lerp < 0:
        lerp = 0
    elif lerp > 1:
        lerp = 1

    x = lerp * dx + a[0]
    y = lerp * dy + a[1]

    _dx = x - p[0]
    _dy = y - p[1]
    square_dist = _dx ** 2 + _dy ** 2
    return np.sqrt(square_dist)
def sgnd_distance(P,V_out,V_in=None):
        value_out=wn_PnPoly(P,V_out)
        dist_out=np.min([distance(V_out[i],V_out[i+1],P) for i in range(len(V_out)-1)])
        if V_in is None:
            if value_out==0:
                return -dist_out
            else:
                return dist_out
        else:
            value_in=wn_PnPoly(P,V_in)
            dist_in=np.min([distance(V_in[i],V_in[i+1],P) for i in range(len(V_in)-1)])
            dist=np.min([dist_in, dist_out])
            if value_out==1 and value_in==0:
                return dist
            else:
                return -dist
            
def plot_matrix(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            plt.scatter(i,j)
            plt.text(i,j,str(A[i,j]))
            
            
            
def plot_results(x, y_pred, y_test):
    x=x.to('cpu')
    y_pred=y_pred.to('cpu')
    y_test=y_test.to('cpu')
    error=torch.linalg.norm(y_test-y_pred)/torch.linalg.norm(y_test)
    fig, ax=plt.subplots(1,2)
    fig.suptitle(f'relative L2 Error: {error:.3e}')
    im0=ax[0].scatter(x[:,0],x[:,1],c=y_test)
    fig.colorbar(im0, ax=ax[0])
    im1=ax[1].scatter(x[:,0],x[:,1],c=y_pred)
    fig.colorbar(im1, ax=ax[1])
    # im2=ax[2].scatter(x,y,c=abs(y_pred-y_test))
    # fig.colorbar(im2, ax=ax[2])
    ax[0].set_title('test')
    ax[1].set_title('pred')
    # ax[2].set_title('error')

        