import numpy as np
from hints2 import *


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

models=[model_mult_4, model_mult_7, model_single_6, model_mult_3, model_mult_8]

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
        np.array([[0,0],[1,0],[1,3/14],[11/14,3/14],[11/14,5/14],[1,5/14],[1,0.5],[11/14,0.5],[11/14,1],[0,1],[0,0]])



    ]

def exp1(poly_out, model,J, N):
    path=Constants.outputs_path+'output1.pt'
    exp3b(model,J=J,N=N, poly_out=poly_out, path=path)
    data=torch.load(path)
    print(np.mean(data['all_iter']))    
    print(np.std(data['all_iter']))     
    print(np.mean(data['all_time'])) 
    print(data['err'][-1]) 


def exp2(poly_out, model,J, N):
    path=Constants.outputs_path+'output1.pt'
    exp3b(model,J=J,N=N, poly_out=poly_out, path=path)
    data=torch.load(path)
    print(np.mean(data['all_iter']))    
    print(np.std(data['all_iter']))     
    print(np.mean(data['all_time'])) 
    print(data['err'][-1]) 

def plot_polygons(fig_path='/Users/idanversano/Documents/project_geo_deeponet/tex/figures/attention_deeponet/'):

    fig, axs = plt.subplots(2, 4, figsize=(7, 7))
    x=np.linspace(0,1,15)
    X,Y=np.meshgrid(x,x,indexing='ij')
    
    vertices=polygons[0]
    axs[0,0].plot(vertices[:, 0], vertices[:, 1], 'b-', marker='o')
    axs[0,0].set_title('Polygon 1')
    

    vertices=polygons[1]
    axs[0,1].plot(vertices[:, 0], vertices[:, 1], 'b-', marker='o')
    axs[0, 1].set_title('Polygon 2')

    vertices=polygons[2]
    axs[0,2].plot(vertices[:, 0], vertices[:, 1], 'b-', marker='o')
    axs[0, 2].set_title('Polygon 3')

    vertices=polygons[3]
    axs[0,3].plot(vertices[:, 0], vertices[:, 1], 'b-', marker='o')
    axs[0, 3].set_title('Polygon 4')
    
    vertices=polygons[4]
    axs[1,0].plot(vertices[:, 0], vertices[:, 1], 'b-', marker='o')
    axs[1, 0].set_title('Polygon 5')
    
    vertices=polygons[5]
    axs[1,1].plot(vertices[:, 0], vertices[:, 1], 'b-', marker='o')
    axs[1, 1].set_title('Polygon 6')
    axs[1,1].scatter(X,Y, color='black',s=5)
    
    vertices=polygons[6]
    axs[1,2].plot(vertices[:, 0], vertices[:, 1], 'b-', marker='o')
    axs[1, 2].set_title('Polygon 7')
    axs[1,2].scatter(X,Y, color='black',s=5)
    
    vertices=polygons[7]
    axs[1,3].plot(vertices[:, 0], vertices[:, 1], 'b-', marker='o')
    axs[1, 3].set_title('Polygon 8')
    axs[1,3].scatter(X,Y, color='black',s=5)
    # plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.2)  # Optional: to fill the polygon with color

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(False)
    plt.tight_layout()  # Adjust subplot spacing

    plt.savefig(fig_path+'fig_polygons.eps', format='eps', bbox_inches='tight', dpi=600)
    plt.show()
# plot_polygons()    
    
exp1(polygons[0],models[3],40,57)
exp1(polygons[1],models[4],20,57)