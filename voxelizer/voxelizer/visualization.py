import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from matplotlib.widgets import Slider, Button

class Image3DPlotCalbacks:
    
    def __init__(self,dataset,idx=0):
        """ 
        Construtor.
        """
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1,2,1,projection='3d')
        self.ax2 = self.fig.add_subplot(1,2,2)

        self.hf = h5py.File(dataset_file,'r')

        self.idx = idx
        self.plane = 15        
        self.axis = 1
        
        self.xlabel = 'Axis X'
        self.ylabel = 'Axis Z' 
        
        self.image,self.label = self.load_image(self.idx) 
        self.colors = np.empty(self.image.shape, dtype=object)
        self.colors[:] = 'blue'
        self.colors[15][15][15] = 'red'
        
        self.DIM = self.image.shape[0]
        self.xx,self.yy = np.meshgrid(range(self.DIM+1),range(self.DIM+1))
        
        self.plot()
    
    def load_image(self,idx):
        """ 
        Load 3d image.
        """
        voxels = self.hf['voxels'][idx]
        label = self.hf['labels'][idx]
        return voxels[...,0],label

    def next(self,event):
        """ 
        Next 3d image.
        """
        self.idx += 1
        self.voxels,self.label = self.load_voxel(self.idx)        
        self.plot()

    def prev(self, event):
        """ 
        Prev 3d image.
        """        
        self.idx -= 1
        self.voxels,self.label = self.load_voxel(self.idx)        
        self.plot()     
        
    def slider(self,val):
        """ 
        Callback for slider panel.
        """
        self.plane = int(val)    
        self.plot()
        
    def plot(self):
        """ 
        Update the plot.
        """
        if self.axis == 0:
            image = self.voxels[self.plane,:,:]
            self.xlabel = 'Axis Y'
            self.ylabel = 'Axis Z'
        elif self.axis == 1:
            image = self.voxels[:,self.plane,:]
            self.xlabel = 'Axis X'
            self.ylabel = 'Axis Z'
        elif self.axis == 2:
            image = self.voxels[:,:,self.plane]
            self.xlabel = 'Axis X'
            self.ylabel = 'Axis Y'
        # Voxel
        self.ax1.clear()
        self.ax1.voxels(self.voxels,facecolors=self.colors,edgecolor='k')        
        self.ax1.set_title('VOXEL IDX: {} LABEL: {}'.format(self.idx,self.label),fontsize=15)

        # Box
        DIM = self.DIM
        self.ax1.plot([0,DIM,DIM,0,0],[0,0,DIM,DIM,0],[0,0,0,0,0],'k')
        self.ax1.plot([0,DIM,DIM,0,0],[0,0,DIM,DIM,0],[DIM,DIM,DIM,DIM,DIM],'k')
        self.ax1.plot([0,DIM,DIM,0,0],[0,0,0,0,0],[0,0,DIM,DIM,0],'k')
        self.ax1.plot([0,DIM,DIM,0,0],[DIM,DIM,DIM,DIM,DIM],[0,0,DIM,DIM,0],'k')        
        
        # Imagem de corte
        self.ax2.clear()
        image = np.rot90(image,k=1) 
        img = self.ax2.imshow(
            (image==0).astype(int),
            origin = 'upper',
            extent = (0,image.shape[0],0,image.shape[1]),
            cmap = 'gray'
        )
        img.set_clim(0, 1)
        self.ax2.set_xlim([-2,33])
        self.ax2.set_ylim([-2,33])
        self.ax2.set_xlabel(self.xlabel,fontsize=12)
        self.ax2.set_ylabel(self.ylabel,fontsize=12)
        
        # Grid
        for i in range(self.xx.shape[0]):
            self.ax2.plot(self.xx[i,:],self.yy[i,:],color='k',linewidth=1)
        for j in range(self.xx.shape[1]):
            self.ax2.plot(self.xx[:,j],self.yy[:,j],color='k',linewidth=1)
            
class GridView:
        
    def __init__(self,data_reader=None,ndim=None,num_figs=None):
        """ 
        Construtor. 
        Ùltima modificação: 16/07/2021
        
        Args:
            data_reader:
            ndim:
            num_figs:
        """
        if data_reader!=None:
            self.data_reader = data_reader
            self.limits = self.data_reader.properties_info['limits']
            
            if ndim==None:
                self.ndim = self.data_reader.properties_info['dimensions']
            else:
                self.ndim = ndim
        else:
            if ndim is None:
                ndim = 2
            self.ndim = ndim
            if self.ndim==2:
                self.limits = np.asarray([0,1,0,1])
            elif self.ndim==3:
                self.limits = np.asarray([0,1,0,1,0,1])
        
        if self.ndim==2:
            if num_figs!=None:
                self.fig,self.ax = [],[]
                for i in range(num_figs):
                    fig,ax = plt.subplots()
                    self.fig.append(fig)
                    self.ax.append(ax)
            else:
                self.fig,self.ax = plt.subplots()
                self.ax.set_xlim(self.limits[0],self.limits[1])
                self.ax.set_ylim(self.limits[2],self.limits[3])
            
        elif self.ndim==3:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection='3d')            
            self.ax.set_xlim(self.limits[0],self.limits[1])
            self.ax.set_ylim(self.limits[2],self.limits[3])
            self.ax.set_zlim(self.limits[4],self.limits[5])
            
                
    def scatter(self,points,labels=None,labels_name=None,title=0,clear_axes=True):
        """ 
        Plota pontos.
        Última modificação: 02/09/2021
        
        Args:
            particles:
            labels:
            labels_name:
            ax:    
        """
        # Clear the current axes.
        if clear_axes:
            self.ax.cla()
        # Plota os pontos
        if labels is None:
            if self.ndim==2:
                self.ax.scatter(points[:,0],points[:,1],s=20)
            elif self.ndim==3:
                self.ax.scatter(points[:,0],points[:,1],points[:,2],s=10)
        else:
            for lb in np.unique(labels):
                self.ax.scatter(points[labels==lb,0],points[labels==lb,1],s=10)
        if labels_name is not None:
            plt.legend(labels_name)
        self.ax.set_title(title)
        # Define os limites dos eixos
        plt.xlim(self.limits[0])
        plt.ylim(self.limits[1])
        if self.ndim==3:
            plt.zlim(self.limits[2])
        plt.axis('equal')
        plt.pause(0.1)
            
    def draw_grid(self,x,y,linewidth=1,color='k'):
        """ 
        Desenha um grid.
        Última modificação: 02/06/2021.
        
        Args:
            x:     coordenadas x do grid.
            y:     coordenadas y do grid.
            color:  cor de desenho do grid.      
        """
        m,n = y.shape[0],x.shape[0]
        xx,yy = np.meshgrid(x,y)
        for i in range(m):
            self.ax.plot(xx[i,:],yy[i,:],color=color,linewidth=linewidth)
        for j in range(n):
            self.ax.plot(xx[:,j],yy[:,j],color=color,linewidth=linewidth)
        plt.axis('equal')    

    def draw_images_dataset(
        self,dataset,key_dense=None,keys_sparse=None
    ):
        """
        Desenha as imagens 2D de um dataset.
        Última atualização: 16/02/2022.

        Args:
            dataset:
            key_dense:
            keys_sparse:

        """
        if len(dataset.shape)==4: # dataset de imagens 2d
            for image in dataset:
                self.draw_image(image)
                plt.pause(1)

        elif len(dataset.shape)==5: # dataset de imagens 3d
            callbacks = Image3DPlotCalbacks(dataset,idx=0)

            # Slider
            label_slider = 'Axis Y'
            plane_ax = plt.axes([0.595, 0.03, 0.35, 0.02])
            plane_slider = Slider(
                ax = plane_ax,
                label = label_slider,
                valmin = 0,
                valmax = 30,
                valinit = callbacks.plane,
                valstep = 1,
                orientation = "horizontal"
            )    
            plane_slider.on_changed(callbacks.slider)

            # Button prev
            axprev = plt.axes([0.14, 0.02, 0.1, 0.05])
            bprev = Button(axprev, 'Previous')
            bprev.on_clicked(callbacks.prev)

            # Button next
            axnext = plt.axes([0.25, 0.02, 0.1, 0.05])
            bnext = Button(axnext, 'Next')
            bnext.on_clicked(callbacks.next)

            #plt.subplots_adjust(bottom=0.1)
            callbacks.fig.tight_layout()
            plt.show()             

    def draw_image(
        self,image,position=[0,0],length=[1.0,1.0],
        frame_res=0,show_frame=False,show_grid=0
    ):
        """
        Desenha uma imagem.
        Última modificação: 16/02/2022.
        
        Args:
            image:      uma matriz contendo uma imagem.                     
            position:   posição espacial imagem.
            length:     comprimento espacial imagem.
            frame_res:  resolução da moldura da imagem.
            show_frame: mostra a moldura da imagem.
            show_grid:  0 - nenhum grid, 1 - grid principal, 2 - grid completo
        """
        img_res = image.shape[0]
        x = np.linspace(position[0],position[0]+length[0],img_res+1)
        y = np.linspace(position[1],position[1]+length[1],img_res+1)        
                        
        # Corrige a orientação
        image = np.rot90(image,k=1).astype(float)
        
        if show_frame:
            self.ax.imshow(
                (image==0).astype(int),
                origin='upper',
                extent=(x[0],x[-1],y[0],y[-1]),
                cmap='gray'
            )
        else:
            self.ax.imshow(
                image[frame_res:img_res-frame_res,frame_res:img_res-frame_res,:],
                origin='lower',
                extent=(x[frame_res],x[-frame_res-1],y[frame_res],y[-frame_res-1])
            )
        # Grid
        if show_grid==1:
            self.draw_grid(x[frame_res:-frame_res],y[frame_res:-frame_res],color='k')
        elif show_grid==2:
            self.draw_grid(x,y,color='k')
        
        plt.xlim(self.limits[0],self.limits[1])
        plt.ylim(self.limits[2],self.limits[3])
        plt.pause(0.1)
                        
    def draw_rectangles(self,left_bottom_coords,width,height,linewidth=2,linecolor='r'): 
        """ 
        Desenha retângulos.
        Última modificação: 04/09/2021.
        
        Args:
            left_bottom_coords:
            width:
            height:
            linewidth: 
            linecolor:                 
        """
        for i in range(left_bottom_coords.shape[0]):
            rect = Rectangle([left_bottom_coords[i,0],left_bottom_coords[i,1]],width,height,
                             linewidth=linewidth,edgecolor=linecolor,facecolor='none')            
            self.ax.add_patch(rect)
        self.ax.axis('equal')

    def draw_circles(self,center_coords,radius,linewidth=2,linecolor='b'):
        """ 
        Desenha retângulos.
        Última modificação: 02/06/2021.
        
        Args:
            center_coords:
            radius:
            linewidth: 
            linecolor: 
                
        """
        for i in range(center_coords.shape[0]):
            circle = Circle(center_coords[i],radius,edgecolor=linecolor,fill=False)
            self.ax.add_patch(circle)
        self.ax.axis('equal')
        
    def draw_cell_cover(self,cell_cover_flat,pgrid,grid_id,linecolor='b',draw_index=False):
        """ 
        Desenha uma cobertura de células.
        Útima modificação: 04/08/2021.
        
        Args:
            cell_cover_flat:
            pgrid:
            grid_id:
            linecolor:
            draw_index:
            
        """
        cell_cover = pgrid.flat_to_ij(cell_cover_flat,grid_id=grid_id)
        cj,ci = cell_cover[1,:],cell_cover[0,:]
        left_bottom_coords = np.asarray([pgrid.y[grid_id][cj],
                                         pgrid.x[grid_id][ci]]).T
        self.draw_rectangles(left_bottom_coords,
                             pgrid.dx[grid_id],
                             pgrid.dy[grid_id],
                             linecolor=linecolor)
        
        if draw_index:
            self.draw_text(left_bottom_coords[:,0],
                           left_bottom_coords[:,1],
                           [str(x) for x in cell_cover_flat],
                           fontsize=10)
            
        self.ax.axis('equal')

    def draw_text(self,x,y,text,fontsize=12):
        """ 
        Plota um vetor de string.
        Última atualização: 05/08/2021.
        
        Args:
            x:
            y:
            text:
        """
        for i in range(len(text)):
            self.ax.text(x[i],y[i],text[i],fontsize=fontsize)

    def draw_predictions(self):
        pass
        # Predictions
        # input_img = np.concatenate(
        #     (images_nn[k:k+1,:,:,0:1],
        #     np.ones((1,self.win_res,self.win_res,1))),axis=3
        # )
        # pred_img = classifier.predict(input_img)
        # pred_img_draw = np.concatenate((pred_img[0,:,:,1:2],pred_img[0,:,:,0:1],
        #                                 np.zeros((self.win_res,self.win_res,1))),axis=2)          
                        
        # ci,cj = centers[k,0],centers[k,1]
        # self.plot.draw_image(
        #     pred_img_draw,
        #     position=[self.x[cj-self.R],self.y[ci-self.R]],
        #     length=[self.x[cj+self.R+1]-self.x[cj-self.R],
        #     self.y[ci+self.R+1]-self.y[ci-self.R]],                                             
        #     frame_res=self.f_res,
        #     show_frame=False,
        #     show_grid=0
        # )            
    

    