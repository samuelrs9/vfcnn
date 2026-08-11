import time
import numpy as np
from scipy import sparse
import tensorflow as tf
from sklearn.neighbors import KDTree

from plots import Plots2D
         
class MultiSparseVoxelizer:
    
    def __init__(self,points,res,limits=None,sim_reader=None,
                 activate_plot=False,img_res=63,f_res=9,grid_types=[0]):
        """ 
        Construtor.
        Última modificação: 04/09/2021.
        
        Args:
            points:
            limits:         limites do grid.
            res:            resolução numérica.
            sim_reader:
            activate_plot:
        """
        self.points = points
        self.res = res
        self.ngrids = len(res)
        self.grid_coords = []
        self.global_coords = []

        self.eps = 1e-12        
        self.ndim = points.shape[1]
        self.sizes = np.zeros((self.ngrids,self.ndim),dtype=int)
        
        if limits is None:
            limits = np.array([points.min(axis=0),points.max(axis=0)]).T            
            
        # Atualiza os limites
        max_res = max(res)   
        limits = limits + [-max_res,max_res]
        num_parts = ((limits[:,1] - limits[:,0])/max_res).astype(int)
        limits[:,1] = limits[:,0] + (num_parts + 1) * max_res
            
        self.indices = np.arange(points.shape[0])   
        self.grid_types = grid_types
                
        for i in range(self.num_grids):
            cells,x,y,nx,ny,dx,dy = self.build_grid_v2(self.data,self.limits,self.res[i],grid_type=grid_types[i])
           
            #### 
            self.sizes[i],self.ny[i] = nx,ny
            self.dx[i],self.dy[i] = dx,dy
            self.x.append(x)
            self.y.append(y)
            self.cells.append(cells)
        
        self.img_res = img_res
        self.f_res = f_res
        self.num_channels = 2
        
        self.stride = self.img_res - 2*self.f_res
        self.r = np.floor(0.5*(self.img_res - 2*self.f_res)).astype(int)
        self.R = np.floor(0.5*self.img_res).astype(int)

        # self.ci_start = np.floor(0.5*self.win_res + 1).astype(int)
        # self.cj_start = np.floor(0.5*self.win_res + 1).astype(int)
        # self.ci_end = self.ny - self.win_res
        # self.cj_end = self.nx - self.win_res
        
        self.activate_plot = activate_plot
        if self.activate_plot:
            if sim_reader is None:
                self.plot = Plots2D(dimensions=2)
            else:
                self.plot = Plots2D(sim_reader)
                        
    
    def build_grid(self,points,limits,res,grid_type=1):
        """ 
        Carrega os pontos em um grid esparso.
        Última modificação: 04/09/2021.
        
        Args:
            points:
            limits:
            res:
            grid_type:
                
        Return:
            depende dos argumentos de entrada.
        """
        size = 1 + ((self.limits[:,1]-self.limits[:,0])/res).astype(int)
        global_coord = [np.linspace(self.limits[i,0],self.limits[i,1],self.size[i]) 
                             for i in range(self.ndim)]
        origin_coord = [self.global_coord[i][0] for i in range(self.ndim)]
        grid_coord_flat = self.compute_coordinates(points,origin_coord,output_flat=True)
        
        if grid_type==0:
            return grid_coord_flat,global_coord,size
        else:
            grid_argsort = grid_coord_flat.argsort()
            grid_sort = grid_coord_flat[grid_argsort]
            
            non_empty_cells,non_empty_cells_index = np.unique(grid_sort,return_index=True)        
            points_in_cell = np.asarray(np.split(grid_argsort,non_empty_cells_index[1:]),dtype=object)                                                    
            
            if grid_type==1:                
                points_in_cell = points_in_cell
                return points_in_cell,grid_coord_flat,global_coord,size
            
            elif grid_type==2:
                non_empty_cells = dict(zip(non_empty_cells,np.arange(non_empty_cells.shape[0])))
                return non_empty_cells,points_in_cell,grid_coord_flat,global_coord,size
    

        
    def extract_images_ap2_v1(self,points_idx,classifier=None):
        """ 
        Extrai imagens de vizinhança usando uma janela que varre o domínio do
        problema.
        Última modificação: 15/06/2021.
        
        Args:
            points_idx: 
            classifier:
                
        Returns:
            
        """ 
        dataset = {}
        particles_idx = []
        flat_target_idx = []
        
        t0,t1,t2,t3 = 0,0,0,0
        #count = 0
                
        grid_cells = self.cells[2][points_idx]
        #grid_cells = self.flat_to_ij(flat_index,grid=0)

        
        grid_cells_unique = np.unique(grid_cells)        
        grid_ij_unique = self.flat_to_ij(grid_cells_unique,grid=2)
        
        grid_cover = np.asarray([self.x[2][grid_ij_unique[1,:]],
                                 self.y[2][grid_ij_unique[0,:]]])
        
        t = time.time()
        neighbors = self.find_square_neighborhood_v2(grid_cells_unique,origin=2,target=1,shift=1,size=7,
                               input_flat=True,output_flat=False,output_refined_grid=True)
                
        t0 = time.time() - t
        
        #centers = grid_ij_unique.T + [self.r,self.r]

        if classifier != None:
            self.plot.ax.scatter(self.data[points_idx,0],self.data[points_idx,1],c='b')
                    
            self.plot.draw_rectangles(grid_cover.T,self.dx[2],self.dy[2],
                                      linewidth=2,linecolor='k')
                
        images_nn = np.empty((grid_cells_unique.shape[0],self.img_res,self.img_res,self.num_channels))
        
        #t = time.time()        
        flat_index = self.find_match_coordinates(grid_cells_unique,origin=2,target=1,
                                                 input_flat=True,output_flat=True)
        
        left_bottom_r = self.find_match_coordinates(grid_cells_unique,origin=2,target=0,
                                                    input_flat=True,output_flat=False)        
        left_bottom_R = left_bottom_r - np.asarray([[self.f_res],[self.f_res]])
        
        #t1 = time.time()-t
        
        #t = time.time()
        for k in range(neighbors.shape[0]):
            # Índices dos pontos na célula
            p_idx = grid_cells==grid_cells_unique[k]
            
            # Índices de pontos vizinhas
            neighbors_coord_R = neighbors[k]            
            neighbors_coord_r = self.flat_to_ij(self.cells[0][points_idx[p_idx]],grid=0)
            
            # Pontos em coordenadas do grid 
            #neighbors_coord_R = self.grid_coordinates[neighbors_R,:]
            #neighbors_coord_r = grid_coord[p_idx,:]
            
            #t = time.time()
            # Cria os canais da imagem            
                        
            flat_channel_1 = self.grid_coord_to_image_coord(neighbors_coord_R,
                                                            left_bottom_R[:,k:k+1],
                                                            return_flat_coord=True)
            np.put(images_nn[k,:,:,0],flat_channel_1,np.ones((flat_channel_1.shape[0])))
                        
            flat_channel_2 = self.grid_coord_to_image_coord(neighbors_coord_r,
                                                            left_bottom_r[:,k:k+1],
                                                            return_flat_coord=True)
            np.put(images_nn[k,:,:,1],flat_channel_2,np.ones((flat_channel_2.shape[0])))
            
            #t2 += time.time() - t
                        
            #t = time.time()
            
            # Dados auxiliares sobre as imagens criadas
            particles_idx.append(points_idx[p_idx])
            flat_target_idx.append(flat_channel_2)
            
            #t3 += time.time() - t
            
            # Predictions
            if classifier != None:
                input_img = np.concatenate((images_nn[k:k+1,:,:,0:1],np.ones(
                                            (1,self.img_res,self.img_res,1))),axis=3)                            
                pred_img = classifier.predict(input_img)
                pred_img_draw = np.concatenate((pred_img[0,:,:,1:2],pred_img[0,:,:,0:1],
                                                np.zeros((self.img_res,self.img_res,1))),axis=2)
                                
                self.plot.draw_image(pred_img_draw,
                                     position=[self.x[0][left_bottom_R[1,k]],self.y[0][left_bottom_R[0,k]]],
                                     length=[self.img_res*self.dx[0],self.img_res*self.dy[0]],
                                     frame_res=self.f_res,
                                     show_frame=False,
                                     show_grid=0)
        
        #t2 = time.time()-t
        
        dataset['particle_idx'] = particles_idx
        dataset['flat_target_idx'] = flat_target_idx
        dataset['input_nn'] = images_nn
        
        # print('t0: {:.4f}'.format(t0))
        # print('t1: {:.4f}'.format(t1))
        # print('t2: {:.4f}'.format(t2))
        #print('t3: {:.4f}'.format(t3))        
        # print('count: {}'.format(count))
        
        return dataset
    

    def extract_images_ap2_v2(self,points_idx,classifier=None):
        """ 
        Extrai imagens de vizinhança usando uma janela que varre o domínio do
        problema.
        Última modificação: 22/06/2021.
        
        Args:
            points_idx: 
            classifier:
                
        Returns:
            
        """ 
        dataset = {}
        particles_idx = []
        flat_target_idx = []
        
        #t0,t1,t2,t3 = 0,0,0,0
        #count = 0
                
        grid_cells = self.cells[2]['cell_by_point'][points_idx]
        #grid_cells = self.flat_to_ij(flat_index,grid=0)

        
        grid_cells_unique = np.unique(grid_cells)        
        grid_ij_unique = self.flat_to_ij(grid_cells_unique,grid=2)
        
        grid_cover = np.asarray([self.x[2][grid_ij_unique[1,:]],
                                 self.y[2][grid_ij_unique[0,:]]])
        
        #t = time.time()
        neighbors = self.find_square_neighborhood_v3(grid_cells_unique,origin=2,target=1,shift=1,size=7,
                               input_flat=True,output_flat=False,output_refined_grid=True)
                
        #t0 = time.time() - t
        
        #centers = grid_ij_unique.T + [self.r,self.r]

        if classifier != None:
            self.plot.ax.scatter(self.data[points_idx,0],self.data[points_idx,1],c='b')
                    
            self.plot.draw_rectangles(grid_cover.T,self.dx[2],self.dy[2],
                                      linewidth=2,linecolor='k')
                
        images_nn = np.empty((grid_cells_unique.shape[0],self.img_res,self.img_res,self.num_channels))
        
        #t = time.time()        
        flat_index = self.find_match_coordinates(grid_cells_unique,origin=2,target=1,
                                                 input_flat=True,output_flat=True)
        
        left_bottom_r = self.find_match_coordinates(grid_cells_unique,origin=2,target=0,
                                                    input_flat=True,output_flat=False)        
        left_bottom_R = left_bottom_r - np.asarray([[self.f_res],[self.f_res]])
        
        #t1 = time.time()-t
        
        #t = time.time()
        for k in range(neighbors.shape[0]):
            # Índices dos pontos na célula
            p_idx = grid_cells==grid_cells_unique[k]
            
            # Índices de pontos vizinhas
            neighbors_coord_R = neighbors[k]            
            neighbors_coord_r = self.flat_to_ij(self.cells[0][points_idx[p_idx]],grid=0)
            
            # Pontos em coordenadas do grid 
            #neighbors_coord_R = self.grid_coordinates[neighbors_R,:]
            #neighbors_coord_r = grid_coord[p_idx,:]
            
            #t = time.time()
            # Cria os canais da imagem            
                        
            flat_channel_1 = self.grid_coord_to_image_coord(neighbors_coord_R,
                                                            left_bottom_R[:,k:k+1],
                                                            return_flat_coord=True)
            np.put(images_nn[k,:,:,0],flat_channel_1,np.ones((flat_channel_1.shape[0])))
                        
            flat_channel_2 = self.grid_coord_to_image_coord(neighbors_coord_r,
                                                            left_bottom_r[:,k:k+1],
                                                            return_flat_coord=True)
            np.put(images_nn[k,:,:,1],flat_channel_2,np.ones((flat_channel_2.shape[0])))
            
            #t2 += time.time() - t
                        
            #t = time.time()
            
            # Dados auxiliares sobre as imagens criadas
            particles_idx.append(points_idx[p_idx])
            flat_target_idx.append(flat_channel_2)
            
            #t3 += time.time() - t
            
            # Predictions
            if classifier != None:
                input_img = np.concatenate((images_nn[k:k+1,:,:,0:1],np.ones(
                                            (1,self.img_res,self.img_res,1))),axis=3)                            
                pred_img = classifier.predict(input_img)
                pred_img_draw = np.concatenate((pred_img[0,:,:,1:2],pred_img[0,:,:,0:1],
                                                np.zeros((self.img_res,self.img_res,1))),axis=2)
                                
                self.plot.draw_image(pred_img_draw,
                                     position=[self.x[0][left_bottom_R[1,k]],self.y[0][left_bottom_R[0,k]]],
                                     length=[self.img_res*self.dx[0],self.img_res*self.dy[0]],
                                     frame_res=self.f_res,
                                     show_frame=False,
                                     show_grid=0)
        
        #t2 = time.time()-t
        
        dataset['particle_idx'] = particles_idx
        dataset['flat_target_idx'] = flat_target_idx
        dataset['input_nn'] = images_nn
        
        #print('t0: {:.4f}'.format(t0))
        #print('t1: {:.4f}'.format(t1))
        #print('t2: {:.4f}'.format(t2))
        #print('t3: {:.4f}'.format(t3))        
        # print('count: {}'.format(count))
        
        return dataset


    def extract_images_ap2_v3(self,points_idx,classifier=None):
        """ 
        Extrai imagens de vizinhança usando uma janela que varre o domínio do
        problema.
        Última modificação: 22/06/2021.
        
        Args:
            points_idx: 
            classifier:
                
        Returns:
            
        """ 
        dataset = {}
        particles_idx = []
        flat_target_idx = []
        
        #t0,t1,t2,t3 = 0,0,0,0
        #count = 0
                
        grid_cells = self.cells[2]['cell_by_point'][points_idx]
        #grid_cells = self.flat_to_ij(flat_index,grid=0)

        
        grid_cells_unique = np.unique(grid_cells)   
        grid_ij_unique = self.flat_to_ij(grid_cells_unique,grid=2)
        
        grid_cover = np.asarray([self.x[2][grid_ij_unique[1,:]],
                                 self.y[2][grid_ij_unique[0,:]]])
        
        t = time.time()
        neighbors = self.find_square_neighborhood_v3(grid_cells_unique,origin=2,target=1,shift=1,size=7,
                               input_flat=True,output_flat=False,output_refined_grid=True)
                
        t0 = time.time() - t
        
        #centers = grid_ij_unique.T + [self.r,self.r]

        if classifier != None:
            self.plot.ax.scatter(self.data[points_idx,0],self.data[points_idx,1],c='b')
                    
            self.plot.draw_rectangles(grid_cover.T,self.dx[2],self.dy[2],
                                      linewidth=2,linecolor='k')
                
        images_nn = np.empty((grid_cells_unique.shape[0],self.img_res,self.img_res,self.num_channels))
        
        t = time.time()        
        flat_index = self.find_match_coordinates(grid_cells_unique,origin=2,target=1,
                                                 input_flat=True,output_flat=True)
        
        left_bottom_r = self.find_match_coordinates(grid_cells_unique,origin=2,target=0,
                                                    input_flat=True,output_flat=False)        
        left_bottom_R = left_bottom_r - np.asarray([[self.f_res],[self.f_res]])
        
        t1 = time.time()-t
        
        t = time.time()
        
        particles_idx = np.zeros(neighbors.shape[0],dtype=object)
        flat_target_idx = np.empty(neighbors.shape[0],dtype=object)
        for k in range(neighbors.shape[0]):
            # Índices dos pontos indefinidos na célula
            p_idx = points_idx[grid_cells==grid_cells_unique[k]]
            
            # Índices de pontos vizinhas
            neighbors_coord_R = neighbors[k]
            neighbors_coord_r = self.flat_to_ij(self.cells[0][p_idx],grid=0)
                        
            #t = time.time()
            # Cria os canais da imagem                                    
            flat_channel_1 = self.grid_coord_to_image_coord(neighbors_coord_R,
                                                            left_bottom_R[:,k:k+1],
                                                            return_flat_coord=True)
            np.put(images_nn[k,:,:,0],flat_channel_1,np.ones((flat_channel_1.shape[0])))
                        
            flat_channel_2 = self.grid_coord_to_image_coord(neighbors_coord_r,
                                                            left_bottom_r[:,k:k+1],
                                                            return_flat_coord=True)
            np.put(images_nn[k,:,:,1],flat_channel_2,np.ones((flat_channel_2.shape[0])))
            
            #t2 += time.time() - t
                        
            #t = time.time()
            
            # Dados auxiliares sobre as imagens
            particles_idx[k] = p_idx
            flat_target_idx[k] = flat_channel_2
            #t3 += time.time() - t
            
            # Predictions
            if classifier != None:
                input_img = np.concatenate((images_nn[k:k+1,:,:,0:1],np.ones(
                                            (1,self.img_res,self.img_res,1))),axis=3)                            
                pred_img = classifier.predict(input_img)
                pred_img_draw = np.concatenate((pred_img[0,:,:,1:2],pred_img[0,:,:,0:1],
                                                np.zeros((self.img_res,self.img_res,1))),axis=2)
                                
                self.plot.draw_image(pred_img_draw,
                                     position=[self.x[0][left_bottom_R[1,k]],self.y[0][left_bottom_R[0,k]]],
                                     length=[self.img_res*self.dx[0],self.img_res*self.dy[0]],
                                     frame_res=self.f_res,
                                     show_frame=False,
                                     show_grid=0)
        
        t2 = time.time()-t
        
        dataset['particle_idx'] = particles_idx
        dataset['flat_target_idx'] = flat_target_idx
        dataset['input_nn'] = images_nn
        
        print('t0: {:.4f}'.format(t0))
        print('t1: {:.4f}'.format(t1))
        print('t2: {:.4f}'.format(t2))
        #print('t3: {:.4f}'.format(t3))        
        # print('count: {}'.format(count))
        
        return dataset

    
    def extract_images_ap2_v4(self,points_idx,classifier=None):
        """ 
        Extrai imagens de vizinhança usando uma janela que varre o domínio do
        problema.
        Última modificação: 23/06/2021.
        
        Args:
            points_idx: 
            classifier:
                
        Returns:
            
        """ 
        dataset = {}
        particles_idx = []
        flat_target_idx = []
        
        #t0,t1,t2,t3 = 0,0,0,0
        #count = 0
                
        grid_cells = self.cells[2]['cell_by_point'][points_idx]
        #grid_cells = self.flat_to_ij(flat_index,grid=0)

        
        grid_cells_unique = np.unique(grid_cells)   
        grid_ij_unique = self.flat_to_ij(grid_cells_unique,grid=2)
        
        grid_cover = np.asarray([self.x[2][grid_ij_unique[1,:]],
                                 self.y[2][grid_ij_unique[0,:]]])
        
        #t = time.time()
        neighbors = self.find_square_neighborhood_v3(grid_cells_unique,origin=2,target=1,shift=1,size=7,
                               input_flat=True,output_flat=False,output_refined_grid=True)
                
        #t0 = time.time() - t
        
        #centers = grid_ij_unique.T + [self.r,self.r]

        if classifier != None:
            self.plot.ax.scatter(self.data[points_idx,0],self.data[points_idx,1],c='b')
                    
            self.plot.draw_rectangles(grid_cover.T,self.dx[2],self.dy[2],
                                      linewidth=2,linecolor='k')
                
        images_nn = np.empty((grid_cells_unique.shape[0],self.img_res,self.img_res,self.num_channels))
        
        #t = time.time()        
        flat_index = self.find_match_coordinates(grid_cells_unique,origin=2,target=1,
                                                 input_flat=True,output_flat=True)
        
        left_bottom_r = self.find_match_coordinates(grid_cells_unique,origin=2,target=0,
                                                    input_flat=True,output_flat=False)        
        left_bottom_R = left_bottom_r - np.asarray([[self.f_res],[self.f_res]])
        
        #t1 = time.time()-t
        
        #t = time.time()
        
        grid_cells_argsort = grid_cells.argsort()        
        grid_cells_sort = grid_cells[grid_cells_argsort]
        points_idx = points_idx[grid_cells_argsort]
    
        _,cells_unique_index= np.unique(grid_cells_sort,return_index=True)    
        
        particles_idx = np.asarray(np.split(points_idx,cells_unique_index[1:]),dtype=object)            
        
        flat_target_idx = np.empty(particles_idx.shape[0],dtype=object)
        #t2 = time.time()-t
        
        #t3 = 0
        #t = time.time()
        for k in range(particles_idx.shape[0]):
            # Índices dos pontos indefinidos na célula
            p_idx = particles_idx[k]
            
            # Índices de pontos vizinhas
            neighbors_coord_R = neighbors[k]
            neighbors_coord_r = self.flat_to_ij(self.cells[0][p_idx],grid=0)
                        
            #t = time.time()
            # Cria os canais da imagem                                    
            flat_channel_1 = self.grid_coord_to_image_coord(neighbors_coord_R,
                                                            left_bottom_R[:,k:k+1],
                                                            return_flat_coord=True)
            np.put(images_nn[k,:,:,0],flat_channel_1,np.ones((flat_channel_1.shape[0])))
                        
            flat_channel_2 = self.grid_coord_to_image_coord(neighbors_coord_r,
                                                            left_bottom_r[:,k:k+1],
                                                            return_flat_coord=True)
            np.put(images_nn[k,:,:,1],flat_channel_2,np.ones((flat_channel_2.shape[0])))
            
            #t3 += time.time() - t
                        
            #t = time.time()
            
            # Dados auxiliares sobre as imagens
            #particles_idx[k] = p_idx
            flat_target_idx[k] = flat_channel_2
            #t3 += time.time() - t
            
            # Predictions
            if classifier != None:
                input_img = np.concatenate((images_nn[k:k+1,:,:,0:1],np.ones(
                                            (1,self.img_res,self.img_res,1))),axis=3)                            
                pred_img = classifier.predict(input_img)
                pred_img_draw = np.concatenate((pred_img[0,:,:,1:2],pred_img[0,:,:,0:1],
                                                np.zeros((self.img_res,self.img_res,1))),axis=2)
                                
                self.plot.draw_image(pred_img_draw,
                                     position=[self.x[0][left_bottom_R[1,k]],self.y[0][left_bottom_R[0,k]]],
                                     length=[self.img_res*self.dx[0],self.img_res*self.dy[0]],
                                     frame_res=self.f_res,
                                     show_frame=False,
                                     show_grid=0)
        
        #t3 = time.time()-t
        
        dataset['particle_idx'] = particles_idx
        dataset['flat_target_idx'] = flat_target_idx
        dataset['input_nn'] = images_nn
        
        # print('t0: {:.4f}'.format(t0))
        # print('t1: {:.4f}'.format(t1))
        # print('t2: {:.4f}'.format(t2))
        # print('t3: {:.4f}'.format(t3))        
        # print('count: {}'.format(count))
        
        return dataset


    def compute_coordinates(self,points_coord,origin_coord,output_flat=False):
        """
        Dado um conjunto de pontos retorna as suas coordenadas de grid.
        Última atualização: 04/09/2021.
        
        Args:
            points_coord: coordenadas dos pontos.
            origin_coord: 
                output_flat
        
        Returns:
           grid_coord: coordenadas dos pontos no grid.
        """
        grid_coord = [np.floor((points_coord[:,i]-origin_coord[i])/
                     self.res).astype(int) for i in range(self.ndim)]
        if output_flat:
            return self.ij_to_flat(np.asarray(grid_coord))
        else:         
            return np.asarray(grid_coord).transpose()


    def compute_coordinates_v1(self,px,py,x0,y0,nx,ny,res,output_flat=False):
        """
        Dado um conjunto de pontos de coordenadas (px,py) retorna um grid esparso.
        Última modificação: 09/06/2021
        
        Args:
            px:  coordenadas x dos pontos.
            py:  coordenadas y dos pontos.
            x0:
            y0:
            nx:
            ny:
            res:    
            output_flat:
        
        Returns:
            coordenadas das células no grid.
        """
        cell_j = np.floor((px - x0 + self.eps)/res).astype(int)
        cell_i = np.floor((py - y0 + self.eps)/res).astype(int)
        
        cell_ij = np.asarray([cell_i,cell_j])
        
        if output_flat:
            return self.ij_to_flat(cell_ij,ni=ny,nj=nx)
        else:            
            return cell_ij

    def compute_coordinates_v2(self,px,py,x,y,nx,ny,res,output_flat=False):
        """
        Dado um conjunto de pontos de coordenadas (px,py) retorna um grid esparso.
        Última modificação: 21/06/2021
        
        Args:
            px:  coordenadas x dos pontos.
            py:  coordenadas y dos pontos.
            x:
            y:
            nx:
            ny:
            res:    
            output_flat:
        
        Returns:
            coordenadas das células no grid.
        """
        x0 = np.asarray([xg[0] for xg in x]).reshape(-1,1)
        y0 = np.asarray([yg[0] for yg in y]).reshape(-1,1)
        
        px = px.reshape(1,-1)
        py = py.reshape(1,-1)
        
        res = np.asarray(res).reshape(-1,1)
        
        cell_j = np.floor((px - x0 + self.eps)/res).astype(int)
        cell_i = np.floor((py - y0 + self.eps)/res).astype(int)
        
        cell_ij = [np.asarray([cell_i[k,:],cell_j[k,:]]) 
                   for k in range(self.num_grids)]
        
        if output_flat:
            cell_flat = [self.ij_to_flat(cell_ij[k],ni=ny[k],nj=nx[k]) 
                         for k in range(self.num_grids)]
            return cell_flat
        else:            
            return cell_ij


    def cell_cover(self,points_idx,grid_id=0,output_flat=True):
        """
        Encontra as células de cobertura de um subconjunto de pontos.
        Última atualização: 02/08/2021.
        
        Args:
            points_idx:
            grid_id:
            output_flat:
        Returns: 
            grid_cover:
        """
        cover_grid_flat = self.cells[grid_id]['cell_by_point'][points_idx]        
        cover_grid_flat_unique = np.unique(cover_grid_flat)        
        if output_flat:
            return cover_grid_flat_unique
        else:
            return self.flat_to_ij(cover_grid_flat_unique,grid_id=grid_id)

        
    def ij_to_flat(self,cell_ij,ni=None,nj=None,grid_id=None):
        """ 
        Faz a conversão de indexação mutidimensional para indexação flat. 
        Última modificação: 09/06/2021.
        
        Args:
            cell_ij:
            ni:
            nj:
            grid_id:
            
        Return:
            flat index.
        """     
        if grid_id != None:
            return np.ravel_multi_index(cell_ij,(self.ny[grid_id],self.nx[grid_id]))
        else:
            return np.ravel_multi_index(cell_ij,(ni,nj))
    
    def flat_to_ij(self,flat_index,ni=None,nj=None,grid_id=None):
        """ 
        Faz a conversão de indexação flat para indexação multidimensional.
        Última modificação: 09/06/2021.
        
        Args:
            cell_ij:
            ni:
            nj:
            grid_id:
            
        Return:
            indexação multidimensional.
        """        
        if grid_id != None:            
            return np.asarray(np.unravel_index(flat_index,(self.ny[grid_id],self.nx[grid_id])))
        else:
            return np.asarray(np.unravel_index(flat_index,(ni,nj)))


    def find_match_coordinates(self,input_index,origin=0,target=0,input_flat=True,output_flat=True):
        """ 
        Faz a conversão entre as indexações dos grids.
        Última modificação: 09/06/2021.
       
        Args:
            input_index:
            origin:
            target:
            input_flat:
            output_flat:
                      
        Return:
           
        """
        if input_flat:
            cell_ij = self.flat_to_ij(input_index,grid=origin)
        else:
            cell_ij = input_index            
        
        px = self.x[origin][cell_ij[1,:]]
        py = self.y[origin][cell_ij[0,:]]

        x0 = self.x[target][0]
        y0 = self.y[target][0]
        
        nx = self.nx[target]
        ny = self.ny[target]
        
        res = self.resolutions[target]
        
        return self.compute_coordinates(px,py,x0,y0,nx,ny,res,output_flat=output_flat)     
       

    def find_square_neighborhood(self,input_index,origin=0,target=0,shift=1,size=5,
                                 input_flat=True,output_flat=True,output_refined_grid=True):
        """ 
        Encontra vizinhança quadrada de uma célula do grid.
        Última modificação: 10/06/2021.
        
        Args:
            
        Returns:
            
        """
        flat_index = self.find_match_coordinates(input_index,origin=2,target=1,
                                            input_flat=True,output_flat=True)
        
        shift_index = flat_index - shift*(self.nx[target] + 1)
        
        target_cells = np.zeros((shift_index.shape[0],4*size-4),dtype=int)
        
        target_index = np.empty(shift_index.shape[0],dtype=object)
        
        for i in range(shift_index.shape[0]):
            # bottom target cells
            target_cells[i,0:size] = np.arange(shift_index[i],shift_index[i] + size,1)
            # left target cells
            target_cells[i,size:2*size-2] = np.arange(shift_index[i] + self.nx[target],
                                          shift_index[i] + (size-2)*self.nx[target] + 1,
                                          self.nx[target])
            # right target cells
            target_cells[i,2*size-2:3*size-4] = target_cells[i,size:2*size-2] + size - 1
            # top target cells
            target_cells[i,3*size-4:4*size-4] = target_cells[i,0:size] + (size - 1)*self.nx[target]
        
        
            target_index_0 = self.cells[2] == input_index[i]
            
            target_index_1 = self.cells[1] == target_cells[i].reshape(1,-1).T
            target_index_1 = target_index_1.sum(axis=0).astype(bool)
            
            if output_refined_grid:
                if output_flat:
                    target_index[i] = self.cells[0][np.logical_or(target_index_0,target_index_1)]
                else:
                    target_flat = self.cells[0][np.logical_or(target_index_0,target_index_1)]
                    target_index[i] = self.flat_to_ij(target_flat,grid=0)
            else:
                target_index[i] = self.indices[np.logical_or(target_index_0,target_index_1)]
        
        return target_index


    def find_square_neighborhood_v2(self,input_index,origin=0,target=0,shift=1,size=5,
                                 input_flat=True,output_flat=True,output_refined_grid=True):
        """ 
        Encontra vizinhança quadrada de uma célula do grid.
        Última modificação: 15/06/2021.
        
        Args:
            
        Returns:
            
        """
        #t = time.time()
        flat_index = self.find_match_coordinates(input_index,origin=2,target=1,
                                                 input_flat=True,output_flat=True)
        #t0 = time.time()
        
        shift_index = flat_index - shift*(self.nx[target] + 1)
        
        target_cells = np.zeros((shift_index.shape[0],4*size-4),dtype=int)
        
        target_index = np.empty(shift_index.shape[0],dtype=object)
        
        #t = time.time()
        #t1_1,t1_2 = 0,0
        for i in range(shift_index.shape[0]):
            # bottom target cells
            target_cells[i,0:size] = np.arange(shift_index[i],shift_index[i] + size,1)
            # left target cells
            target_cells[i,size:2*size-2] = np.arange(shift_index[i] + self.nx[target],
                                          shift_index[i] + (size-2)*self.nx[target] + 1,
                                          self.nx[target])
            # right target cells
            target_cells[i,2*size-2:3*size-4] = target_cells[i,size:2*size-2] + size - 1
            # top target cells
            target_cells[i,3*size-4:4*size-4] = target_cells[i,0:size] + (size - 1)*self.nx[target]
        
            #t = time.time()
            target_index_0 = self.cells[2] == input_index[i] #np.equal(self.cells[2],input_index[i]) 
            #t1_1 += time.time()-t
            
            #t = time.time()
            target_index_1 = self.cells[1] == target_cells[i].reshape(1,-1).T # np.equal(self.cells[1],target_cells[i].reshape(1,-1).T)
            target_index_1 = np.logical_or.reduce(target_index_1) #target_index_1.sum(axis=0).astype(bool)
            #t1_2 += time.time()-t
            
            if output_refined_grid:
                if output_flat:
                    target_index[i] = self.cells[0][np.logical_or(target_index_0,target_index_1)]
                else:
                    target_flat = self.cells[0][np.logical_or(target_index_0,target_index_1)]
                    target_index[i] = self.flat_to_ij(target_flat,grid=0)
            else:
                target_index[i] = self.indices[np.logical_or(target_index_0,target_index_1)]
        
        #t1 = time.time()-t
        
        #print('t0: {:.4f}'.format(t0))
        #print('t1: {:.4f}'.format(t1))
        #print('t1_1: {:.4f} \t t1_2: {:.4f}'.format(t1_1,t1_2))
        #print('t2: {:.4f}'.format(t2))
        
        return target_index

    def find_square_neighborhood_v3(self,input_index,origin=0,target=0,shift=1,size=5,
                                 input_flat=True,output_flat=True,output_refined_grid=True):
        """ 
        Encontra vizinhança quadrada de uma célula do grid.
        Última modificação: 15/06/2021.
        
        Args:
            
        Returns:
            
        """
        #t = time.time()
        flat_index = self.find_match_coordinates(input_index,origin=2,target=1,
                                                 input_flat=True,output_flat=True)
        #t0 = time.time()-t
        
        shift_index = flat_index - shift*(self.nx[target] + 1)
        
        target_cells = np.zeros((shift_index.shape[0],4*size-4),dtype=int)
        
        target_index = np.empty(shift_index.shape[0],dtype=object)
        
        #t = time.time()
        #t1_1,t1_2 = 0,0
        for i in range(shift_index.shape[0]):
            # bottom target cells
            target_cells[i,0:size] = np.arange(shift_index[i],shift_index[i] + size,1)
            # left target cells
            target_cells[i,size:2*size-2] = np.arange(shift_index[i] + self.nx[target],
                                          shift_index[i] + (size-2)*self.nx[target] + 1,
                                          self.nx[target])
            # right target cells
            target_cells[i,2*size-2:3*size-4] = target_cells[i,size:2*size-2] + size - 1
            # top target cells
            target_cells[i,3*size-4:4*size-4] = target_cells[i,0:size] + (size - 1)*self.nx[target]
        
            #t = time.time()
            target_cell_0 = self.cells[2]['non_empty_cells'][input_index[i]]
            target_index_0 = self.cells[2]['points_by_cell'][target_cell_0]
            #t1_1 += time.time()-t
            
            #t = time.time()
            target_cell_1 = [self.cells[1]['non_empty_cells'].get(t) for t in target_cells[i] 
                             if self.cells[1]['non_empty_cells'].get(t) != None]
            ind = self.cells[1]['points_by_cell'][target_cell_1]
            if ind.shape[0]>0:
                target_index_1 = np.hstack(ind)
            else:
                target_index_1 = np.empty(0,dtype=int)
            #t1_2 += time.time()-t
            
            if output_refined_grid:
                if output_flat:
                    target_index[i] = self.cells[0][np.hstack((target_index_0,target_index_1))]
                else:
                    target_flat = self.cells[0][np.hstack((target_index_0,target_index_1))]
                    target_index[i] = self.flat_to_ij(target_flat,grid=0)
            else:
                target_index[i] = np.hstack((target_index_0,target_index_1))
        
        #t1 = time.time()-t
        
        #print('t0: {:.4f}'.format(t0))
        #print('t1: {:.4f}'.format(t1))
        #print('t1_1: {:.4f} \t t1_2: {:.4f}'.format(t1_1,t1_2))
        #print('t2: {:.4f}'.format(t2))
        
        return target_index

    def find_square_neighborhood_v4(self,input_index,origin=0,target=0,shift=1,size=5,
                                 input_flat=True,output_flat=True,output_refined_grid=True):
        """ 
        Encontra vizinhança quadrada de uma célula do grid.
        Última modificação: 15/06/2021.
        
        Args:
            
        Returns:
            
        """
        #t = time.time()
        flat_index = self.find_match_coordinates(input_index,origin=2,target=1,
                                                 input_flat=True,output_flat=True)
        #t0 = time.time()-t
        
        shift_index = flat_index - shift*(self.nx[target] + 1)
        
        target_cells = np.zeros((shift_index.shape[0],4*size-4),dtype=int)
        
        target_index = np.empty(shift_index.shape[0],dtype=object)
        
        #t1_i = time.time()
        #t1_0,t1_1,t1_2 = 0,0,0
        for i in range(shift_index.shape[0]):
            #t = time.time()
            # bottom target cells
            target_cells[i,0:size] = np.arange(shift_index[i],shift_index[i] + size,1)
            # left target cells
            target_cells[i,size:2*size-2] = np.arange(shift_index[i] + self.nx[target],
                                          shift_index[i] + (size-2)*self.nx[target] + 1,
                                          self.nx[target])
            # right target cells
            target_cells[i,2*size-2:3*size-4] = target_cells[i,size:2*size-2] + size - 1
            # top target cells
            target_cells[i,3*size-4:4*size-4] = target_cells[i,0:size] + (size - 1)*self.nx[target]
                
            #t1_0 += time.time()-t
        
            #t = time.time()
            target_cell_0 = self.cells[2]['non_empty_cells'][input_index[i]]
            target_index_0 = self.cells[2]['points_by_cell'][target_cell_0]
            #t1_1 += time.time()-t
            
            #t = time.time()
            target_cell_1 = [self.cells[1]['non_empty_cells'].get(k) for k in target_cells[i] 
                             if self.cells[1]['non_empty_cells'].get(k) != None]
            ind = self.cells[1]['points_by_cell'][target_cell_1]
            
            if ind.shape[0]>0:
                target_index_1 =  np.concatenate(ind.tolist())
            else:
                target_index_1 = np.empty(0,dtype=int)
            #t1_2 += time.time()-t
            
            if output_refined_grid:
                if output_flat:
                    target_index[i] = self.cells[0][np.concatenate([target_index_0,target_index_1])]
                else:
                    target_flat = self.cells[0][np.concatenate([target_index_0,target_index_1])]
                    target_index[i] = self.flat_to_ij(target_flat,grid=0)
            else:
                target_index[i] = np.concatenate([target_index_0,target_index_1])
        
        #t1 = time.time()-t1_i
        
        #print('t0: {:.12f}'.format(t0))
        #print('t1: {:.12f}'.format(t1))
        #print('t1_0: {:.12f} \t t1_1: {:.12f} \t t1_2: {:.12f}'.format(t1_0,t1_1,t1_2))
        #print('t2: {:.4f}'.format(t2))
        
        return target_index

        
    def find_nonempty_cell_neighborhood(self,cell_input,grid_id=1,return_count=False):
        """ 
        Encontra vizinhança quadrada de uma célula do grid.
        Última modificação: 04/08/2021.
        
        Args:
            cell_input:
            grid_id:
            return_count:
            
        Returns:
            
        """
        #t1 = time.time()
        cell_neighborhood = np.zeros((cell_input.shape[0],9),dtype=int)
        nonempty_cell_neighborhood = np.zeros(cell_input.shape[0],dtype=object)
        
        if return_count:
            nonempty_cell_neighborhood_count = np.zeros(len(cell_input),dtype=int)        
            
        #t1_0,t1_1 = 0,0,0
        for i in range(cell_input.shape[0]):
            #t = time.time()
            # ci,cj = np.ogrid[-self.nx[grid_id]:self.nx[grid_id]+1:self.nx[grid_id],
            #                cell_input[i]-1:cell_input[i]+2]
            # cell_neighborhood[i,:] = (ci+cj).flatten()
            
            # middle cells                        
            cell_neighborhood[i,3:6] = np.asarray([cell_input[i]-1,cell_input[i],cell_input[i]+1])
            # bottom cells
            cell_neighborhood[i,0:3] = cell_neighborhood[i,3:6] - self.nx[grid_id]
            # top cells
            cell_neighborhood[i,6:9] = cell_neighborhood[i,3:6] + self.nx[grid_id]
                
            #t1_0 += time.time()-t
            
            #t = time.time()
            nonempty_cell_neighborhood[i] = np.asarray([self.cells[grid_id]['non_empty_cells'].get(k) for k in cell_neighborhood[i] 
                                             if self.cells[grid_id]['non_empty_cells'].get(k) != None])
            if return_count:
                nonempty_cell_neighborhood_count[i] = nonempty_cell_neighborhood[i].shape[0]
            
            #t1_1 += time.time()-t
            
        #t1 = time.time()-t1
        
        #print('t1: {:.12f}'.format(t1))
        #print('t1_0: {:.12f} \t t1_1: {:.12f}'.format(t1_0,t1_1))
        if return_count:
            return nonempty_cell_neighborhood,nonempty_cell_neighborhood_count
        else:
            return nonempty_cell_neighborhood

        
    def grid_coord_to_image_coord(self,grid_coord,left_bottom_coord,return_flat_coord=False):
        """ 
        Última modificação: 11/06/2021.
        
        Args:
            grid_coord:         pontos em coordenadas de grid.
            left_bottom_coord:  centro em coordenadas de grid.
            return_flat_coord: 
            
        Returns:
            
        """
        # Converte para coordenadas locais de imagem
        image_coord = grid_coord - left_bottom_coord   
        if return_flat_coord:
            # Converte para indexação flat
            flat_image_coord = np.ravel_multi_index(image_coord,(self.img_res,self.img_res))         
            return flat_image_coord
        else:
            return image_coord
        