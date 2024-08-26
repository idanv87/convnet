for v in dir():
    exec('del '+ v)
    del v


from matplotlib.ticker import ScalarFormatter

import time
import math

from scipy.sparse import csr_array, tril
import scipy.sparse
import scipy.sparse.linalg
from scipy.stats import qmc
import scipy
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from memory_profiler import memory_usage, profile

import sys


from utils import upsample, fft, solve_gmres, track, find_sub_indices, GS_method, generate_polygon_vertices, plot_matrix, sgnd_distance
from constants import Constants
from utils import  grf, evaluate_model, generate_grf, np_wn, SelfAttention2, generate_polygon_vertices

from two_d_data_set import *
from packages.my_packages import Gauss_zeidel, interpolation_2D, gs_new, Gauss_zeidel2

# from two_d_model import  deeponet
import  more_models_2, more_models_3, more_models_4, more_models_5, more_models_6, more_models_8
from test_deeponet import domain
from main import generate_f_g

from df_polygon import generate_example, generate_rect, generate_rect2, generate_example_2, generate_obstacle, generate_obstacle2, generate_two_obstacles, generate_example3, make_domain, make_obstacle
# 2024.08.14.08.35.39best_model.pth several domains more_models2
# 2024.08.14.07.35.15best_model.pth single domain more_models_2
# 2024.08.20.04.03.38best_model.pth models_3 several no mask

class IterationCounter:
    def __init__(self):
        self.num_NN_iterations = 0
        self.num_GS_iterations = 0
        self.num_gmres_iterations = 0

class TimeCounter:
    def __init__(self):
        self.num_NN = 0
        self.num_GS = 0
        self.num_gmres = 0

model_single=more_models_2.deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.08.14.07.35.15best_model.pth', map_location=torch.device('cpu'))
model_single.load_state_dict(best_model['model_state_dict'])

model_mult=more_models_2.deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.08.14.08.35.39best_model.pth', map_location=torch.device('cpu'))
model_mult.load_state_dict(best_model['model_state_dict'])

model_mult_3=more_models_3.deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.08.20.04.03.38best_model.pth', map_location=torch.device('cpu'))
model_mult_3.load_state_dict(best_model['model_state_dict'])


model_mult_4=more_models_4.deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.08.20.08.55.58best_model.pth', map_location=torch.device('cpu'))
model_mult_4.load_state_dict(best_model['model_state_dict'])

model_mult_5=more_models_5.deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.08.26.04.32.13best_model.pth', map_location=torch.device('cpu'))
model_mult_5.load_state_dict(best_model['model_state_dict'])

model_single_6=more_models_6.vannila_deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.08.26.10.35.07best_model.pth', map_location=torch.device('cpu'))
model_single_6.load_state_dict(best_model['model_state_dict'])


model_mult_8=more_models_8.deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.08.26.11.31.36best_model.pth', map_location=torch.device('cpu'))
model_mult_8.load_state_dict(best_model['model_state_dict'])

def conv_NN(int_points,F,dom,mask,sgnd,model):
    sgnd=torch.tensor(sgnd.flatten(), dtype=torch.float32)  
    y=torch.tensor(int_points,dtype=torch.float32)
    f=torch.tensor(F,dtype=torch.float32)
    with torch.no_grad():
        pred2=model.forward2([y,f.unsqueeze(0),dom.unsqueeze(0),mask.unsqueeze(0),sgnd.unsqueeze(0)])
    return torch.real(pred2).numpy()+1J*torch.imag(pred2).numpy()

def hints(A,b,x0, J, alpha,X,Y,X_ref,Y_ref,dom,mask, valid_indices, model, good_indices, poly_out, poly_in):
    iter_counter=IterationCounter()
    
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    sgnd= np.zeros((Constants.n,Constants.n))
    for i in range(Constants.n):
        for j in range(Constants.n):
            sgnd[i,j]=sgnd_distance((d_ref.x[i],d_ref.y[j]),poly_out,poly_in)
          
    
    sgnd_y=np.array([sgnd_distance((X[i],Y[i]),poly_out,poly_in) for i in range(len(X))])
    int_points=np.concatenate((np.vstack([X,Y]).T,sgnd_y.reshape(sgnd_y.shape[0], 1)), axis=1)

    A=csr_array(A)
    L=csr_array(tril(A,k=0))
    U=A-L   
    err=[]
    color=[]
    start=time.time()
    for k in range(12000):
      
        if (k+1)%J==0:
        
            iter_counter.num_NN_iterations+=1

            f_real=(-A@x0+b).real[good_indices]
            f_imag=(-A@x0+b).imag[good_indices]

            s_real=np.std(f_real)/alpha
            s_imag=np.std(f_imag)/alpha
            f_ref_real=np.zeros(Constants.n**2)
            f_ref_imag=np.zeros(Constants.n**2)
            
            f_ref_real[valid_indices]=(f_real)/(s_real+1e-15)
            f_ref_imag[valid_indices]=(f_imag)/(s_imag+1e-15)

            corr_real=(conv_NN(int_points,f_ref_real,dom,mask,sgnd, model))*s_real
            iter_counter.num_NN_iterations+=1

        
            corr_imag=(conv_NN(int_points,f_ref_imag,dom,mask,sgnd, model))*s_imag
            iter_counter.num_NN_iterations+=1
            corr=corr_real+1J*corr_imag
            x0=x0+corr
  
            
        else:
            # start=time.time()  
            x0=Gauss_zeidel2(U,L,b,x0)
            iter_counter.num_gmres_iterations+=1
            # x0,_,iter,_=solve_gmres(A,b,x0,maxiter=30, tol=1e-10)

        if k %50 ==0:   
            pass 
            print(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
            
            print(k)

        err.append(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
        if err[-1]<1e-10 or err[-1]>100:

            time_counter=time.time()-start
            print(f' hints took: {time.time()-start} with {k} iteration and with error {err[-1]}')
            break
             

    return err, color, J, alpha,k, iter_counter, time_counter 


def exp3b(model, sigma=0.1,l=0.2,mean=0):
    poly_in=None
    # poly_out=np.array([[0,0],[1,0],[1,4/14],[4/14,4/14],[4/14,1],[0,1],[0,0]])
    poly_out=np.array([[0,0],[0.5,0],[0.5,0.5],[1,0.5],[1,1],[0,1],[0,0]])
    hight=12/14
    poly_out=np.array([[0,0],[1,0],[1,0.5],[9/14,0.5],[9/14,hight],[7/14,hight],
                       [7/14,0.5],[5/14,0.5],[5/14,hight],[3/14,hight],[3/14,0.5],
                       [0,0.5],[0,0]])
    # poly_out,A, dom,mask, X,Y, X_ref, Y_ref, valid_indices=torch.load(Constants.outputs_path+'two_stripes.pt')
    # poly_out=np.array([[0,0],[1,0],[1,3/14],[3/14,3/14],[3/14,5/14],[1,5/14],[1,1],[0,1],[0,0]])
    # poly_out=np.array([[0,0],[1,0],[1,5/14],[9/14,5/14],[9/14,1],[4/14,1],[4/14,5/14],[0,5/14],[0,0]])
    # poly_out=np.array([[2/14,2/14],[10/14,2/14],[10/14,3/14],[4/14,3/14],[4/14,10/14],[2/14,10/14],[2/14,2/14]])-np.array([2/14,2/14])

    A, dom,mask, X,Y, X_ref, Y_ref, valid_indices=make_domain(57,poly_out)
    # poly_out=np.array([[2/14,2/14],[1,2/14],[1,5/14],[9/14,5/14],[9/14,9/14],[4/14,9/14],[4/14,5/14],[2/14,5/14],[2/14,2/14]])
    # poly_out=np.array([[2/14,2/14],[10/14,2/14],[10/14,3/14],[4/14,3/14],[4/14,10/14],[2/14,10/14],[2/14,2/14]])-np.array([2/14,2/14])
    # poly_out=np.array([[0,2/14],[3/14,2/14],[3/14,0],[6/14,0],[6/14,2/14],[8/14,2/14],
                    #    [8/14,5/14],[6/14,5/14],[6/14,8/14],[3/14,8/14],[3/14,5/14],[0,5/14],[0,2/14]])+np.array([7/14,0])
    # poly_out=generate_polygon_vertices(30)-np.array([3/14,3/14])
    # poly_out=np.array([[0,0],[1,0],[1,0.5],[0.5,0.5],[0.5,1],[0,1],[0,0]])

    # torch.save((poly_out,A,dom,mask, X, Y,X_ref,Y_ref, valid_indices), Constants.outputs_path+'two_stripes.pt')
    # A,dom,mask, X,Y, X_ref, Y_ref, valid_indices, poly_out, poly_in=generate_obstacle2(57)
    # plt.scatter(X,Y);plt.show()
    
    

    
    # torch.save((A,dom,mask, X, Y,X_ref,Y_ref, valid_indices), Constants.outputs_path+'L225.pt')
    # A,dom,mask, X, Y,X_ref,Y_ref, valid_indices=torch.load(Constants.outputs_path+'L225.pt')
    
    # A,dom,mask, X,Y, X_ref, Y_ref, valid_indices, poly_out, poly_in=torch.load(Constants.outputs_path+'obs225.pt')

    print(A.shape)
    good_indices=find_sub_indices(X,Y,X_ref,Y_ref)
    # F=generate_grf(X,Y,n_samples=20,sigma=1, l=0.7 ,mean=10)
    f_ref=np.zeros(225)
   
    all_iter=[]
    all_time=[]
    for i in range(1):
        # b=np.cos(6*math.pi*np.array(X))*np.cos(6*math.pi*np.array(Y))
        # b=np.exp(np.array(X)**2)
        b=np.random.normal(10,10,A.shape[0])
        # u=scipy.sparse.linalg.spsolve(A, b)
        f_ref[valid_indices]=b[good_indices]
        
        x0=(b+1J*b)*0.001
        
        # x,err,iters,time_counter=solve_gmres(A,b,x0)
        # print(time_counter)
        # print(iters)
        # print(err)
        
        err, color, J, alpha, iters, iter_counter, time_counter=hints(A,b,x0,J=300, alpha=0.3,X=X,Y=Y,X_ref=X_ref,Y_ref=Y_ref,dom=dom,mask=mask, valid_indices=valid_indices, model=model, good_indices=good_indices, poly_out=poly_out,poly_in=poly_in)  
        all_iter.append(iters)
        all_time.append(time_counter)

    torch.save({'X':X, 'Y':Y,'all_iter':all_iter, 'all_time':all_time,'err':err}, Constants.outputs_path+'output14.pt')     
    
# exp3b(model_mult_8)  
exp3b(model_mult_4)    
# exp3b(model_mult_3) 
# exp3b(model_single_6) 

# data=torch.load(Constants.outputs_path+'output14.pt')
# print(np.mean(data['all_iter']))    
# # print(np.std(data['all_iter']))     
# print(np.mean(data['all_time']))    




















