import numpy as np
import matplotlib.pyplot as plt
from test_deeponet import domain
from utils import *
from scipy.sparse import csr_matrix, kron, identity, lil_matrix
from packages.my_packages import  interpolation_2D, Restriction_matrix
from scipy.sparse import coo_array, bmat
def generate_domains(i1,i2,j1,j2):
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    x_ref=d_ref.x
    y_ref=d_ref.y
    return domain(x_ref[i1:i2], y_ref[j1:j2])

def generate_f_g(shape, seedf):

        f=generate_random_matrix(shape,seed=seedf)
        
        f=(f-np.mean(f))/np.std(f)
        
        
       
        return f
def mask_matrix(valid_indices):
    mask = np.zeros((Constants.n**2,Constants.n**2))
    for i in range(Constants.n**2):
        for j in range(Constants.n**2):
            if ((i in valid_indices) and (j in valid_indices))== False:
                mask[i,j]=-1e20
    return  mask

def masking_coordinates(X,Y):
        xx,yy=np.meshgrid(np.linspace(0,1, Constants.n),np.linspace(0,1, Constants.n),indexing='ij')
        X0=xx.flatten()
        Y0=yy.flatten()
        original_points=[(X0[i],Y0[i]) for i in range(len(X0))]
        points=np.array([(X[i],Y[i]) for i in range(len(X))])
        valid_indices=[]
        masked_indices=[]
        for j,p in enumerate(original_points):
            dist=[np.linalg.norm(np.array(p)-points[i]) for i in range(points.shape[0])]
            if np.min(dist)<1e-14:
                valid_indices.append(j)
            else:
                masked_indices.append(j)    
        return valid_indices, masked_indices     
        
def generate_example(sigma=0.1,l=0.2,mean=0):  
    x1=np.linspace(0,1/2,8) 
    y1=np.linspace(0,1,15)
    X1,Y1=np.meshgrid(x1,y1,indexing='ij')
    X1,Y1=X1.flatten(), Y1.flatten()

    x2=np.linspace(8/14,1,7) 
    y2=np.linspace(0,1/2,8)
    X2,Y2=np.meshgrid(x2,y2,indexing='ij')
    X2,Y2=X2.flatten(), Y2.flatten()
    d1=domain(x1,y1)
    d2=domain(x2,y2)
    D1=d1.D.todense()
    D2=d2.D.todense()
    D=block_matrix(D1,D2)

    # k=0
    # for i in range(len(X1)):
    #     plt.scatter(X1[i],Y1[i])
    #     plt.text(X1[i],Y1[i],str(k))
    #     k+=1
                
    # for i in range(len(X2)):
    #     plt.scatter(X2[i],Y2[i])
    #     plt.text(X2[i],Y2[i],str(k))
    #     k+=1
    # plt.show()    

    intersection_indices_l=[105,106,107,108,109,110,111,112]
    l_jump=-15
    r_jump=15

    dx=(x1[1]-x1[0])
    for c in intersection_indices_l[1:]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
    D[105,105]=-4/dx/dx-2/dx*Constants.l
    D[105,90]=1/dx/dx
    D[105,120]=1/dx/dx
    D[105,106]=2/dx/dx

    intersection_indices_r=[120,121,122,123,124,125,126,127]
    l_jump=-15
    r_jump=8

    dx=(x1[1]-x1[0])
    for c in intersection_indices_r[1:-1]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
        
    D[120,120]=-4/dx/dx-2/dx*Constants.l
    D[120,120+l_jump]=1/dx/dx
    D[120,120+r_jump]=1/dx/dx
    D[120,121]=2/dx/dx

    D[127,127]=-4/dx/dx-2/dx*Constants.l
    D[127,127+l_jump]=1/dx/dx
    D[127,127+r_jump]=1/dx/dx
    D[127,126]=2/dx/dx


    X,Y=np.concatenate([X1,X2]), np.concatenate([Y1,Y2])

    valid_indices, non_valid_indices=masking_coordinates(X, Y)     
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask=mask_matrix(valid_indices)
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)
    
    # f=generate_f_g(len(X), 1)
    f=generate_grf(X,Y,1,sigma,l,mean)[0]

    f_ref[valid_indices]=f
    # f_ref=torch.tensor(f_ref, dtype=torch.float32)
    
    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),dom,mask, X,Y, X,Y,valid_indices

def generate_rect(N):
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)
    # domain with low resolution:
    d=generate_domains(0,8, 0,15)
    X_ref,Y_ref=d.X, d.Y
    mask=mask_matrix(valid_indices)
    mask=torch.tensor(mask, dtype=torch.float32)
    # domain with hugh resolution:
    # N=Constants.n*2-1
    new_domain=domain(np.linspace(d.x[0],d.x[-1],int((N+1)/2)),np.linspace(d.y[0],d.y[-1],N))
    X,Y=new_domain.X, new_domain.Y
    A,G=new_domain.solver()
    #  d.valid_indices are the indices in the reference rectangle which stays inside the new low resolution domain
    
    return A,dom,mask, X, Y,X_ref,Y_ref, d.valid_indices

def generate_rect2(N):
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)
    # domain with low resolution:
    d=generate_domains(0,15, 0,15)
    X_ref,Y_ref=d.X, d.Y
    mask = np.zeros((Constants.n**2,Constants.n**2))
    mask[:, d.non_valid_indices] = float('-inf') 
    # domain with hugh resolution:
    # N=Constants.n*2-1
    new_domain=domain(np.linspace(0,1,N),np.linspace(0,1,N))
    X,Y=new_domain.X, new_domain.Y
    # k=0
    # for i in range(len(X)):
    #     plt.scatter(X[i],Y[i])
    #     plt.text(X[i],Y[i],str(k))
    #     k+=1
    # plt.show()    
    A,G=new_domain.solver(X.reshape((N,N)))
    #  d.valid_indices are the indices in the reference rectangle which stays inside the new low resolution domain
    
    return A,dom,torch.tensor(mask, dtype=torch.float32), X, Y,X_ref,Y_ref, d.valid_indices


def generate_example_2(N=29):  
    # X_ref, Y_ref domain points contained in the referwbce domain
    x1=np.linspace(0,1/2,8) 
    y1=np.linspace(0,1,15)
    x2=np.linspace(8/14,1,7) 
    y2=np.linspace(0,1/2,8)

    d1=domain(x1,y1)
    d2=domain(x2,y2)
    X_ref,Y_ref=np.concatenate([d1.X,d2.X]), np.concatenate([d1.Y,d2.Y])
    
    
    
    # d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    # f_ref=np.zeros(d_ref.nx*d_ref.ny)
    x=np.linspace(0,1,N)
    y=np.linspace(0,1,N)
    x1=np.linspace(x[0],x[int((N-1)/2)],int((N+1)/2)) 
    y1=np.linspace(y[0],y[-1],N)
    X1,Y1=np.meshgrid(x1,y1,indexing='ij')
    X1,Y1=X1.flatten(), Y1.flatten()

    x2=np.linspace(x[int((N+1)/2)],x[-1],int((N-1)/2)) 
    y2=np.linspace(y[0],y[int((N-1)/2)],int((N+1)/2))
    X2,Y2=np.meshgrid(x2,y2,indexing='ij')
    X2,Y2=X2.flatten(), Y2.flatten()
    
    
    

    
    
    d1=domain(x1,y1)
    d2=domain(x2,y2)
    D=lil_matrix(bmat([[d1.D, None], [None, d2.D]]))
    # D1=d1.D.todense()
    # D2=d2.D.todense()
    # D=block_matrix(D1,D2)
    
    
    
    
    # k=0
    # for i in range(len(X1)):
    #     plt.scatter(X1[i],Y1[i])
    #     # plt.text(X1[i],Y1[i],str(k))
    #     k+=1
                
    # for i in range(len(X2)):
    #     plt.scatter(X2[i],Y2[i])
    #     plt.text(X2[i],Y2[i],str(k))
    #     k+=1
    # plt.show()    

    intersection_indices_l=[int((N-1)/2*N)+i for i in range(int((N+1)/2))]
    l_jump=-N
    r_jump=N

    dx=(x1[1]-x1[0])
    for c in intersection_indices_l[1:]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
    D[int((N-1)/2*N),int((N-1)/2*N)]=-4/dx/dx-2/dx*Constants.l
    D[int((N-1)/2*N),int((N-1)/2*N)+N]=1/dx/dx
    D[int((N-1)/2*N),int((N-1)/2*N)-N]=1/dx/dx
    D[int((N-1)/2*N),int((N-1)/2*N)+1]=2/dx/dx

    intersection_indices_r=[int((N+1)/2*N)+i for i in range(int((N+1)/2))]
    l_jump=-N
    r_jump=int((N+1)/2)

    dx=(x1[1]-x1[0])
    for c in intersection_indices_r[1:-1]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
        
    D[int((N+1)/2*N),int((N+1)/2*N)]=-4/dx/dx-2/dx*Constants.l
    D[int((N+1)/2*N),int((N+1)/2*N)+l_jump]=1/dx/dx
    D[int((N+1)/2*N),int((N+1)/2*N)+r_jump]=1/dx/dx
    D[int((N+1)/2*N),int((N+1)/2*N)+1]=2/dx/dx

    p=int((N+1)/2*N)+int((N-1)/2)
    D[p,p]=-4/dx/dx-2/dx*Constants.l
    D[p,p+l_jump]=1/dx/dx
    D[p,p+r_jump]=1/dx/dx
    D[p,p-1]=2/dx/dx

        

    X,Y=np.concatenate([X1,X2]), np.concatenate([Y1,Y2])

    valid_indices, non_valid_indices=masking_coordinates(X, Y)     
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask=mask_matrix(valid_indices)
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)

  
    
    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),dom,mask, X,Y, X_ref, Y_ref, valid_indices


def generate_obstacle():  
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    obs=generate_domains(3,11, 9,12)
    
    
    X_ref=[]
    Y_ref=[]
    good_ind=[]
    for i in range(len(d_ref.X)):
        dist=[abs(d_ref.X[i]-obs.X[j])+abs(d_ref.Y[i]-obs.Y[j]) for j in range(len(obs.X))]
        if np.min(dist)>1e-10:
            X_ref.append(d_ref.X[i])
            Y_ref.append(d_ref.Y[i])
            good_ind.append(i)   
    D=lil_matrix(d_ref.D)[good_ind,:][:,good_ind]
    valid_indices, non_valid_indices=masking_coordinates(X_ref, Y_ref) 
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask=mask_matrix(valid_indices)
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)
    f=generate_f_g(len(X_ref), 1)
    func=interpolation_2D(X_ref,Y_ref,f)
    f_ref[valid_indices]=func(X_ref,Y_ref)
    X=X_ref
    Y=Y_ref
    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices


def generate_obstacle2(N):  
    d_out=domain(np.linspace(0,1,N),np.linspace(0,1,N))
    d0=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    
    
    # obs=domain(d_out.x[int (N/4):int (3*N/4)],d_out.y[int (N/4):int (3*N/4)])
    # obs=domain(d_out.x[int (N/10)+1:int (9*N/10)+1],d_out.y[int(N/10)+1:int (9*N/10)+1])
    obs=domain(d_out.x[int (N/4)+1:int (3*N/4)+1],d_out.y[int(N/2)+1:int (7*N/10)+1])
    
    
    X=[]
    Y=[]
    good_ind=[]
    for i in range(len(d_out.X)):
        dist=[abs(d_out.X[i]-obs.X[j])+abs(d_out.Y[i]-obs.Y[j]) for j in range(len(obs.X))]
        if np.min(dist)>1e-10:
            X.append(d_out.X[i])
            Y.append(d_out.Y[i])
            good_ind.append(i)   
    D=lil_matrix(d_out.D)[good_ind,:][:,good_ind]
    valid_indices, non_valid_indices=masking_coordinates(X, Y) 
    
    f_ref=np.zeros(d0.nx*d0.ny)
    mask=mask_matrix(valid_indices)
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d0.X.reshape(-1, 1), d0.Y.reshape(-1, 1))), dtype=torch.float32)
    

    
    X_ref=[]
    Y_ref=[]
    for i in range(len(d0.X)):
        dist=[abs(d0.X[i]-obs.X[j])+abs(d0.Y[i]-obs.Y[j]) for j in range(len(obs.X))]
        if np.min(dist)>1e-10:
            X_ref.append(d0.X[i])
            Y_ref.append(d0.Y[i])
            

    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),dom,mask, X,Y, X_ref, Y_ref, valid_indices













def generate_two_obstacles(N=29):  
    d0=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    d_out=domain(np.linspace(0,1,N),np.linspace(0,1,N))
    # obs=domain(d_out.x[int (N/4):int (3*N/4)],d_out.y[int (N/4):int (3*N/4)])
    # obs=domain(d_out.x[int (N/10)+1:int (9*N/10)+1],d_out.y[int(N/10)+1:int (9*N/10)+1])
    obs1=domain(d_out.x[int (1*N/5)+1:int (2*N/5)+1],d_out.y[int(2*N/4)+1:int (3*N/4)+1])
    obs2=domain(d_out.x[int (3*N/5)+1:int (4*N/5)+1],d_out.y[int(2*N/4)+1:int (3*N/4)+1])
    
    
    X=[]
    Y=[]
    good_ind=[]
    for i in range(len(d_out.X)):
        dist1=[abs(d_out.X[i]-obs1.X[j])+abs(d_out.Y[i]-obs1.Y[j]) for j in range(len(obs1.X))]
        dist2=[abs(d_out.X[i]-obs2.X[j])+abs(d_out.Y[i]-obs2.Y[j]) for j in range(len(obs2.X))]
        dist=dist1+dist2
        if np.min(dist)>1e-10:
            X.append(d_out.X[i])
            Y.append(d_out.Y[i])
            good_ind.append(i)   
    D=d_out.D.todense()[good_ind,:][:,good_ind]
    valid_indices, non_valid_indices=masking_coordinates(X, Y) 
    
    f_ref=np.zeros(d0.nx*d0.ny)
    mask = np.zeros((len(f_ref),len(f_ref)))
    mask[:, non_valid_indices] = float('-inf')  
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d0.X.reshape(-1, 1), d0.Y.reshape(-1, 1))), dtype=torch.float32)
    


    X_ref=[]
    Y_ref=[]
    for i in range(len(d0.X)):
        dist1=[abs(d0.X[i]-obs1.X[j])+abs(d0.Y[i]-obs1.Y[j]) for j in range(len(obs1.X))]
        dist2=[abs(d0.X[i]-obs2.X[j])+abs(d0.Y[i]-obs2.Y[j]) for j in range(len(obs2.X))]
        dist=dist1+dist2
        if np.min(dist)>1e-10:
            X_ref.append(d0.X[i])
            Y_ref.append(d0.Y[i])
            


    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),dom,mask, X,Y, X_ref, Y_ref, valid_indices


def generate_example3(N):
      # X_ref, Y_ref domain points contained in the referwbce domain
    x0=np.linspace(0,1,Constants.n)
    y0=np.linspace(0,1,Constants.n)
    
    x1=np.linspace(x0[0],x0[3],4) 
    y1=np.linspace(0,1,15)
    x2=np.linspace(x0[4],1,11) 
    y2=np.linspace(0,x0[3],4)

    d1=domain(x1,y1)
    d2=domain(x2,y2)
    X_ref,Y_ref=np.concatenate([d1.X,d2.X]), np.concatenate([d1.Y,d2.Y])
    
    
    
    # d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    # f_ref=np.zeros(d_ref.nx*d_ref.ny)
    m1=int((x0[3]-x0[0])*(N-1)+1)
    m2=N-m1
    x=np.linspace(0,1,N)
    y=np.linspace(0,1,N)
    x1=np.linspace(x[0],x[m1-1],m1) 
    y1=np.linspace(0,1,N)
    X1,Y1=np.meshgrid(x1,y1,indexing='ij')
    X1,Y1=X1.flatten(), Y1.flatten()

    x2=np.linspace(x[m1],1,m2) 
    y2=np.linspace(0,x[m1-1],m1)
    X2,Y2=np.meshgrid(x2,y2,indexing='ij')
    X2,Y2=X2.flatten(), Y2.flatten()
    
    
    

    
    
    d1=domain(x1,y1)
    d2=domain(x2,y2)
    D=lil_matrix(bmat([[d1.D, None], [None, d2.D]]))
    # D1=d1.D.todense()
    # D2=d2.D.todense()
    # D=block_matrix(D1,D2)
    
    
    
    
    # k=0
    # for i in range(len(X1)):
    #     plt.scatter(X1[i],Y1[i])
    #     plt.text(X1[i],Y1[i],str(k))
    #     k+=1
                
    # for i in range(len(X2)):
    #     plt.scatter(X2[i],Y2[i])
    #     plt.text(X2[i],Y2[i],str(k))
    #     k+=1
    # plt.show()    
    
    intersection_indices_l=[int((m1-1)*N)+i for i in range(m1)]
    l_jump=-N
    r_jump=N

    dx=(x1[1]-x1[0])
    for c in intersection_indices_l[1:]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
    D[int((m1-1)*N),int((m1-1)*N)]=-4/dx/dx-2/dx*Constants.l
    D[int((m1-1)*N),int((m1-1)*N)+r_jump]=1/dx/dx
    D[int((m1-1)*N),int((m1-1)*N)+l_jump]=1/dx/dx
    D[int((m1-1)*N),int((m1-1)*N)+1]=2/dx/dx

    intersection_indices_r=[int((m1)*N)+i for i in range(m1)]
    l_jump=-N
    r_jump=m1

    dx=(x1[1]-x1[0])
    for c in intersection_indices_r[1:-1]:
        D[c,c]=-4/dx/dx
        D[c,c-1]=1/dx/dx
        D[c,c+1]=1/dx/dx
        D[c,c+r_jump]=1/dx/dx
        D[c,c+l_jump]=1/dx/dx
        
    D[int((m1)*N),int((m1)*N)]=-4/dx/dx-2/dx*Constants.l
    D[int((m1)*N),int((m1)*N)+l_jump]=1/dx/dx
    D[int((m1)*N),int((m1)*N)+r_jump]=1/dx/dx
    D[int((m1)*N),int((m1)*N)+1]=2/dx/dx

    p=intersection_indices_r[-1]
    D[p,p]=-4/dx/dx-2/dx*Constants.l
    D[p,p+l_jump]=1/dx/dx
    D[p,p+r_jump]=1/dx/dx
    D[p,p-1]=2/dx/dx

        

    X,Y=np.concatenate([X1,X2]), np.concatenate([Y1,Y2])

    valid_indices, non_valid_indices=masking_coordinates(X, Y)     
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask=mask_matrix(valid_indices)
    mask=torch.tensor(mask, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32)

  
    
    return csr_matrix(D)+Constants.k*scipy.sparse.identity(D.shape[0]),dom,mask, X,Y, X_ref, Y_ref, valid_indices


def create_polygon(N,poly):
    x=np.linspace(0,1,N)
    y=np.linspace(0,1,N)
    X,Y=np.meshgrid(x,y,indexing='ij')
    X=X.flatten()
    Y=Y.flatten()
    Dom=[]
    ref_ind=[]
    for k in range(len(X)):
        
        if  wn_PnPoly((X[k],Y[k]),poly):
            i=int(k/N)
            j=int(k-i*N)
            Dom.append([X[k],Y[k]])
            ref_ind.append([i,j])
    
    return Dom, ref_ind
 
def make_domain(N,poly):
   
    Dom_ref, _= create_polygon(Constants.n,poly)
    Dom, ref_ind=create_polygon(N,poly)
    X_ref=[d[0] for d in Dom_ref]
    Y_ref=[d[1] for d in Dom_ref]
    X=[d[0] for d in Dom]
    Y=[d[1] for d in Dom]
    
    h=1/(N-1)
    Dx=lil_matrix(np.zeros((len(Dom),len(Dom))).astype(complex))
    Dy=lil_matrix(np.zeros((len(Dom),len(Dom))).astype(complex))
    x=np.linspace(0,1,N)
    y=np.linspace(0,1,N)

    for k in range(len(Dom)):
        ind=ref_ind[k]
        i=ind[0]
        j=ind[1]

        try:
            left_nbhd_index=Dom.index([x[i-1],y[j]])
            if i>0 and wn_PnPoly((x[i-1],y[j]),poly):
                left_nbhd=1
            else:
                left_nbhd=0    
        except:
            left_nbhd=0    
            
        try:
            right_nbhd_index=Dom.index([x[i+1],y[j]])
            if wn_PnPoly((x[i+1],y[j]),poly):
                right_nbhd=1
            else:
                right_nbhd=0    
        except:
            right_nbhd=0     

        if left_nbhd and right_nbhd:
            Dx[k,k]=-2/h**2
            Dx[k,left_nbhd_index]=1/h**2
            Dx[k,right_nbhd_index]=1/h**2
        else:    
            if left_nbhd:
                Dx[k,left_nbhd_index]=2/(h**2)
                Dx[k,k]=(-2-2*(h*Constants.l))/(h**2)
            if right_nbhd:
                Dx[k,right_nbhd_index]=2/(h**2)
                Dx[k,k]=(-2-2*(h*Constants.l))/(h**2)     
            
        try:
            up_nbhd_index=Dom.index([x[i],y[j-1]])
            if j>0 and wn_PnPoly((x[i],y[j-1]),poly):
                up_nbhd=1
            else:
                up_nbhd=0    
        except:
            up_nbhd=0    
            
        try:
            down_nbhd_index=Dom.index([x[i],y[j+1]])
            if wn_PnPoly((x[i],y[j+1]),poly):
                down_nbhd=1
            else:
                down_nbhd=0    
        except:
            down_nbhd=0     

        if up_nbhd and down_nbhd:
            Dy[k,k]=-2/h**2
            Dy[k,up_nbhd_index]=1/h**2
            Dy[k,down_nbhd_index]=1/h**2
        else:    
            if up_nbhd:
                Dy[k,up_nbhd_index]=2/(h**2)
                Dy[k,k]=(-2-2*(h*Constants.l))/(h**2)
            if down_nbhd:
                Dy[k,down_nbhd_index]=2/(h**2)
                Dy[k,k]=(-2-2*(h*Constants.l))/(h**2) 

    D=Dx+Dy+Constants.k*scipy.sparse.identity(Dx.shape[0])
    valid_indices, non_valid_indices=masking_coordinates(X, Y)     
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask=mask_matrix(valid_indices)
    mask=torch.tensor(mask, dtype=torch.float32)
    
    poly_out=poly
    sgnd= np.zeros((Constants.n,Constants.n))
    for i in range(Constants.n):
        for j in range(Constants.n):
            sgnd[i,j]=sgnd_distance((d_ref.x[i],d_ref.y[j]),poly_out)
    # dom=torch.tensor(sgnd, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32) 

    # k=0
    # for i in range(len(X)):
    #     plt.scatter(X[i],Y[i])
    #     plt.text(X[i],Y[i],str(k))
    #     k+=1
    # plt.show()          
    return csr_matrix(D), dom,mask, X,Y, X_ref, Y_ref, valid_indices

def in_exterior(p,poly):
    value=np.sum([is_between(poly[i],poly[i+1],p) for i in range(len(poly)-1)])
    return wn_PnPoly(p,poly)==0 or value>0

def create_obstacle(N,poly):
    
    x=np.linspace(0,1,N)
    y=np.linspace(0,1,N)
    X,Y=np.meshgrid(x,y,indexing='ij')
    X=X.flatten()
    Y=Y.flatten()
    Dom=[]
    ref_ind=[]
    for k in range(len(X)):
        
        if  in_exterior((X[k],Y[k]),poly):
            i=int(k/N)
            j=int(k-i*N)
            Dom.append([X[k],Y[k]])
            ref_ind.append([i,j])
    
    return Dom, ref_ind


def make_obstacle(N,poly):
   
    Dom_ref, _= create_obstacle(Constants.n,poly)
    Dom, ref_ind=create_obstacle(N,poly)
    X_ref=[d[0] for d in Dom_ref]
    Y_ref=[d[1] for d in Dom_ref]
    X=[d[0] for d in Dom]
    Y=[d[1] for d in Dom]
    
    h=1/(N-1)
    Dx=lil_matrix(np.zeros((len(Dom),len(Dom)))).astype(complex)
    Dy=lil_matrix(np.zeros((len(Dom),len(Dom)))).astype(complex)
    x=np.linspace(0,1,N)
    y=np.linspace(0,1,N)

    for k in range(len(Dom)):
        ind=ref_ind[k]
        i=ind[0]
        j=ind[1]

        try:
            left_nbhd_index=Dom.index([x[i-1],y[j]])
            
            if i>0 and in_exterior((x[i-1],y[j]),poly):
                left_nbhd=1
            else:
                left_nbhd=0    
        except:
            left_nbhd=0    
            
        try:
            right_nbhd_index=Dom.index([x[i+1],y[j]])
            if in_exterior((x[i+1],y[j]),poly):
                right_nbhd=1
            else:
                right_nbhd=0    
        except:
            right_nbhd=0     

        if left_nbhd and right_nbhd:
            Dx[k,k]=-2/h**2
            Dx[k,left_nbhd_index]=1/h**2
            Dx[k,right_nbhd_index]=1/h**2
        else:    
            
            if left_nbhd:
                Dx[k,left_nbhd_index]=2/(h**2)
                Dx[k,k]=(-2-2*(h*Constants.l))/(h**2)
            if right_nbhd:
                Dx[k,right_nbhd_index]=2/(h**2)
                Dx[k,k]=(-2-2*(h*Constants.l))/(h**2)     
            
        try:
            up_nbhd_index=Dom.index([x[i],y[j-1]])
            if j>0 and in_exterior((x[i],y[j-1]),poly):
                up_nbhd=1
            else:
                up_nbhd=0    
        except:
            up_nbhd=0    
            
        try:
            down_nbhd_index=Dom.index([x[i],y[j+1]])
            if in_exterior((x[i],y[j+1]),poly):
                down_nbhd=1
            else:
                down_nbhd=0    
        except:
            down_nbhd=0     

        if up_nbhd and down_nbhd:
            Dy[k,k]=-2/h**2
            Dy[k,up_nbhd_index]=1/h**2
            Dy[k,down_nbhd_index]=1/h**2
        else:    
            if up_nbhd:
                Dy[k,up_nbhd_index]=2/(h**2)
                Dy[k,k]=(-2-2*(h*Constants.l))/(h**2)
            if down_nbhd:
                Dy[k,down_nbhd_index]=2/(h**2)
                Dy[k,k]=(-2-2*(h*Constants.l))/(h**2) 

    D=Dx+Dy+Constants.k*scipy.sparse.identity(Dx.shape[0])
    valid_indices, non_valid_indices=masking_coordinates(X, Y)     
    d_ref=domain(np.linspace(0,1,Constants.n),np.linspace(0,1,Constants.n))
    f_ref=np.zeros(d_ref.nx*d_ref.ny)
    mask=mask_matrix(valid_indices)
    mask=torch.tensor(mask, dtype=torch.float32)
    poly_out=np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
    poly_in=poly
    sgnd= np.zeros((Constants.n,Constants.n))
    for i in range(Constants.n):
        for j in range(Constants.n):
            sgnd[i,j]=sgnd_distance((d_ref.x[i],d_ref.y[j]),poly_out,poly_in)
    # dom=torch.tensor(sgnd, dtype=torch.float32)
    dom=torch.tensor(np.hstack((d_ref.X.reshape(-1, 1), d_ref.Y.reshape(-1, 1))), dtype=torch.float32) 
             
    return csr_matrix(D), dom,mask, X,Y, X_ref, Y_ref, valid_indices


# A1, dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_example_2(15)  
# plt.scatter(X_ref,Y_ref,color='red')

# poly=np.array([[0,0],[1.0001,0.],[1.001,0.5],[0.75,0.5],[0.75,1.0001],[0.25,1.0001],[0.25,0.5],[0,0.5],[0,0]])
# A2, dom,mask, X,Y, X_ref, Y_ref, valid_indices=make_domain(15,poly)
# plt.scatter(X,Y,color='black')
# plt.show()
# A=A1-A2

# print(A2[0,1])

# print(A2.shape)
# 
        
    # k=0
    # for i in range(len(X_ref)):
    #     plt.scatter(X_ref[i],Y_ref[i])
    #     plt.text(X_ref[i],Y_ref[i],str(k))
    #     k+=1
                

    # plt.show()   
    
    

# A,f_ref,f,dom,mask, X,Y, X_ref, Y_ref, valid_indices=generate_example_2()


# generate_rect2(14)     
# D,f,dom,mask=generate_example()

# A, f_ref,f,dom,mask, X, Y, valid_indices =generate_rect2(30)
# print(mask.shape)
# print(f_ref.shape)
