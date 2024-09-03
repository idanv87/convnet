import numpy as np
from hints2 import *
import pandas as pd


def print_with_lines(df, interval=4):
    for i in range(0, len(df), interval):
        print(df.iloc[i:i+interval])
        print('-' * 30)  # This is the horizontal line
# for i,poly_out in enumerate(polygons):
        # print(i)
        # A, dom,mask, X,Y, X_ref, Y_ref, valid_indices=make_domain(225 ,poly_out)
        # torch.save((poly_out,A,dom,mask, X, Y,X_ref,Y_ref, valid_indices), Constants.outputs_path+'polygon'+str(i+1)+'.pt')

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

model_mult_7=more_models_7.deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.08.26.20.04.13best_model.pth', map_location=torch.device('cpu'))
model_mult_7.load_state_dict(best_model['model_state_dict'])



model_mult_8=more_models_8.deeponet(dim=2,f_shape=Constants.n**2, domain_shape=2, p=80) 
best_model=torch.load(Constants.path+'runs/'+'2024.08.26.11.31.36best_model.pth', map_location=torch.device('cpu'))
model_mult_8.load_state_dict(best_model['model_state_dict'])

models=[model_mult_4, model_mult_7, model_single_6, model_mult_3]
        # model_mult_8]

hight=12/14                  
polygons=[np.array([[0,0],[1,0],[1,3/14],[3/14,3/14],[3/14,5/14],[1,5/14],[1,1],[0,1],[0,0]]),
          np.array([[0,0],[1,0],[1,0.5],[9/14,0.5],[9/14,hight],[7/14,hight],
                       [7/14,0.5],[5/14,0.5],[5/14,hight],[3/14,hight],[3/14,0.5],
                       [0,0.5],[0,0]]),
          np.array([[0,0],[1,0],[1,5/14],[9/14,5/14],[9/14,1],[4/14,1],[4/14,5/14],[0,5/14],[0,0]]),
         generate_polygon_vertices(30),
         np.array([[0,0],[1,0],[1,7/14],[7/14,7/14],[7/14,1],[0,1],[0,0]]),
         np.array([[0,0],[1,0],[1,3/14],[0.5,3/14],[0.5,4/14],[1,4/14],[1,0.5],[0.5,0.5],[0.5,1],[0,1],[0,0]]),
         np.array([[0,0],[1,0],[1,3/14],[0.5,3/14],[0.5,5/14],[1,5/14],[1,0.5],[0.5,0.5],[0.5,1],[0,1],[0,0]]),
        np.array([[0,0],[1,0],[1,3/14],[11/14,3/14],[11/14,5/14],[1,5/14],[1,0.5],[11/14,0.5],[11/14,1],[0,1],[0,0]]),
        
        np.array([[0,0],[1,0],[1,3/14],[0.5,3/14],[0.5,5/14],[1,5/14],[1,0.5],[0.5,0.5],[0,1],[0,0]]),
        np.array([[0,0],[10/14,0],[1,0.5],[10/14,10/14],[0,0]]),
        np.array([[0,0],[1,0],[5/14,1],[0,0]]),
        np.array([[0,0],[1,0],[1,7/14],[10/14,1],[0,1],[0,0]])
        # np.array([[0,0],[10/14,0],[7/14,0.5],[3/14,5/14],[6/14,0.5],[5/14,1],[0,0]])
        
        

]
# plt.plot(polygons[-1][:,0],polygons[-1][:,1])
# plt.show()

def exp1(poly_out, model,J, N):
    path=Constants.outputs_path+'output1.pt'
    exp3b(model,J=J,N=N, poly_out=poly_out, path=path)
    data=torch.load(path)
    print(np.mean(data['all_iter']))    
    print(np.std(data['all_iter']))     
    print(np.mean(data['all_time'])) 
    print(data['err'][-1]) 

def exp2(poly_out, model,J, N):
    path=Constants.outputs_path+'output0.pt'
    exp3b(model,J=J,N=N, poly_out=poly_out, path=path)
    data=torch.load(path)
    print(np.mean(data['all_iter']))    
    print(np.std(data['all_iter']))     
    print(np.max(data['err'])) 
    return np.mean(data['all_iter']), np.std(data['all_iter']), np.max(data['err'])
    
    
# exp1(polygons[5],models[2],50,57)n
data=[]
for i,p in enumerate(polygons):
    for j,m in enumerate(models):
        mean, std,err=exp2(p,m,200,113)
        data.append(('polygon'+str(i+1),'model'+str(j+1),mean,std,err))
torch.save(data,Constants.outputs_path+'output2.pt')

data=torch.load(Constants.outputs_path+'output2.pt')
df = pd.DataFrame(data, columns=['', '', 'mean','std','err'])
print_with_lines(df)

def plot_polygons(fig_path='/Users/idanversano/Documents/project_geo_deeponet/tex/figures/attention_deeponet/'):

    fig, axs = plt.subplots(3, 4, figsize=(10, 5))
    x=np.linspace(0,1,15)
    X,Y=np.meshgrid(x,x,indexing='ij')
    
    vertices=polygons[0]
    axs[0,0].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[0,0].set_title('Polygon 1')
    # axs[1,3].scatter(X,Y, color='black',s=5)
    

    vertices=polygons[1]
    axs[0,1].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[0, 1].set_title('Polygon 2')
    # axs[1,3].scatter(X,Y, color='black',s=5)

    vertices=polygons[2]
    axs[0,2].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[0, 2].set_title('Polygon 3')
    # axs[1,3].scatter(X,Y, color='black',s=5)

    vertices=polygons[3]
    axs[0,3].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[0, 3].set_title('Polygon 4')
    # axs[1,3].scatter(X,Y, color='black',s=5)
    
    vertices=polygons[4]
    axs[1,0].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[1, 0].set_title('Polygon 5')
    # axs[1,3].scatter(X,Y, color='black',s=5)
    
    vertices=polygons[5]
    axs[1,1].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[1, 1].set_title('Polygon 6')
    # axs[1,1].scatter(X,Y, color='black',s=5)
    
    vertices=polygons[6]
    axs[1,2].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[1, 2].set_title('Polygon 7')
    # axs[1,2].scatter(X,Y, color='black',s=5)
    
    vertices=polygons[7]
    axs[1,3].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[1, 3].set_title('Polygon 8')
    
    vertices=polygons[8]
    axs[2,0].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[2, 0].set_title('Polygon 9')
    vertices=polygons[9]
    axs[2,1].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[2, 1].set_title('Polygon 10')
    vertices=polygons[10]
    axs[2,2].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[2, 2].set_title('Polygon 11')
    vertices=polygons[11]
    axs[2,3].plot(vertices[:, 0], vertices[:, 1], 'b-')
    axs[2, 3].set_title('Polygon 12')
    # axs[1,3].scatter(X,Y, color='black',s=5)
    # plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.2)  # Optional: to fill the polygon with color

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(False)
    plt.tight_layout()  # Adjust subplot spacing

    plt.savefig(fig_path+'fig_polygons.eps', format='eps', bbox_inches='tight', dpi=600)
    plt.show()
# plot_polygons()    
    
# exp1(polygons[5],models[-1],40,57)
# exp1(polygons[1],models[4],40,57)
t_polygons=[np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]),
       np.array([[0,0],[0.5,0],[0.5,0.5],[0,0.5],[0,0]]),
       np.array([[0,0],[1,0],[1,0.5],[0,0.5],[0,0]]),
       np.array([[0,0],[0.5,0],[0.5,1],[0,1],[0,0]]),
       np.array([[0.5,0],[1,0],[1,0.5],[0.5,0.5],[0.5,0]]),
       np.array([[0.5,0],[1,0],[1,1],[0.5,1],[0.5,0]]),
       np.array([[0,0.5],[0.5,0.5],[0.5,1],[0,1],[0,0.5]]),
       np.array([[0,0.5],[1,0.5],[1,1],[0,1],[0,0.5]]),
       np.array([[0.5,0.5],[1,0.5],[1,1],[0.5,1],[0.5,0.5]])]
def plot_trained_polygons(fig_path='/Users/idanversano/Documents/project_geo_deeponet/tex/figures/attention_deeponet/'):
    
    fig, axs = plt.subplots(3, 3, figsize=(7, 7))
    custom_xlim = (0-0.1, 1+0.1)
    custom_ylim = (0-0.1, 1+0.1)

# Setting the values for all axes.
    plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
    x=np.linspace(0,1,15)
    X,Y=np.meshgrid(x,x,indexing='ij')
    
    vertices=t_polygons[0]
    axs[0,0].plot(vertices[:, 0], vertices[:, 1], 'b-')
    

    vertices=t_polygons[1]
    axs[0,1].plot(vertices[:, 0], vertices[:, 1], 'b-')


    vertices=t_polygons[2]
    axs[0,2].plot(vertices[:, 0], vertices[:, 1], 'b-')


    vertices=t_polygons[3]
    axs[1,0].plot(vertices[:, 0], vertices[:, 1], 'b-')
    
    vertices=t_polygons[4]
    axs[1,1].plot(vertices[:, 0], vertices[:, 1], 'b-')
    
    vertices=t_polygons[5]
    axs[1,2].plot(vertices[:, 0], vertices[:, 1], 'b-')

    
    vertices=t_polygons[6]
    axs[2,0].plot(vertices[:, 0], vertices[:, 1], 'b-')

    
    vertices=t_polygons[7]
    axs[2,1].plot(vertices[:, 0], vertices[:, 1], 'b-')

    
    vertices=t_polygons[8]
    axs[2,2].plot(vertices[:, 0], vertices[:, 1], 'b-')

    # plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.2)  # Optional: to fill the polygon with color

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(False)
    plt.tight_layout()  # Adjust subplot spacing

    plt.savefig(fig_path+'fig_traines.eps', format='eps', bbox_inches='tight', dpi=600)
    plt.show()
# plot_trained_polygons()

def plot_obstacle(fig_path='/Users/idanversano/Documents/project_geo_deeponet/tex/figures/attention_deeponet/'):
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    custom_xlim = (0-0.1, 1+0.1)
    custom_ylim = (0-0.1, 1+0.1)

# Setting the values for all axes.
    plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
    x=np.linspace(0,1,15)
    X,Y=np.meshgrid(x,x,indexing='ij')
    
    pol1=np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
    pol2=np.array([[0.25,0.25],[0.5,0.25],[0.5,0.5],[0.25,0.5],[0.25,0.25]])
    pol3=np.array([[0.25,0.25],[0.75,0.25],[0.75,0.75],[0.25,0.75],[0.25,0.25]])
    vertices1=pol1
    vertices2=pol2
    vertices3=pol3
    axs[0].plot(vertices1[:, 0], vertices1[:, 1], 'b-')
    axs[0].plot(vertices2[:, 0], vertices2[:, 1], 'b-')
    axs[0].set_title('Obstacle 1')
    
    axs[1].plot(vertices1[:, 0], vertices1[:, 1], 'b-')
    axs[1].plot(vertices3[:, 0], vertices3[:, 1], 'b-')
    axs[1].set_title('Obstacle 2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(False)
    plt.tight_layout()  # Adjust subplot spacing

    plt.savefig(fig_path+'fig_obstacle.eps', format='eps', bbox_inches='tight', dpi=600)
    plt.show()
# plot_obstacle()
    
