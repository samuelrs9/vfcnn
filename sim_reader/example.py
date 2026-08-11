import os
import matplotlib.pyplot as plt

from data import DataReader
    
def view_simulation(sim_config_file):
    """ 
    Visualiza um simulação bidimensional.
    
    Args:        
        initial_step:
        final_step:
        skip_steps:
        pause:
    """ 
    sim_reader = DataReader(sim_config_file)
    
    initial_step = sim_reader.data_info['initial_step']
    final_step = sim_reader.data_info['final_step']
    lim = sim_reader.properties_info['limits']
    dt = sim_reader.properties_info['dt']

    # Coordenadas dos pontos da bound box
    bx = [lim[0][0],lim[0][1],lim[0][1],lim[0][0],lim[0][0]]
    by = [lim[1][0],lim[1][0],lim[1][1],lim[1][1],lim[1][0]]

    plt.figure()
    plt.pause(2)

    for step in range(initial_step,final_step+1):
        plt.cla()
        
        # Carrega partículas do passo atual
        particles = sim_reader.get_step(step)
        
        # Plota partículas
        plt.scatter(particles[:,0],particles[:,1],c='b',edgecolors='k')
        plt.plot(bx,by,'-k',linewidth=4,)
        plt.title("frame {}".format(step))        
        #plt.xlim([lim[0][0]-0.1,lim[0][1]+0.1])
        #plt.ylim([lim[1][0]-0.1,lim[1][1]+0.1])
        plt.axis('equal')
        plt.axis('off') 
        
        plt.pause(dt)

if __name__=='__main__':
    current_dir = os.path.dirname(__file__)
    config_file = os.path.join(current_dir,'example','sim_config.yaml')
    view_simulation(config_file)