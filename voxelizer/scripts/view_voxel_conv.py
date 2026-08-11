""" 
Visualizador de voxels
Última modificação: 06/09/2021.
"""
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

working_dir = "e:\\doc\\bpartcnn\\"
data_dir = working_dir + 'data\\dambreak3d\\'

trainset_file = data_dir + 'approach1\\train_31_4.13_2.hdf5'
valset_file = data_dir + 'approach1\\val_31_4.13_2.hdf5'

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1,projection='3d')
ax2 = fig.add_subplot(1,2,2)

# Plot de voxel com callbacks
class VoxelPlotCalbacks:
    
    def __init__(self,dataset_file,idx=0):
        self.hf = h5py.File(dataset_file,'r')

        self.idx = idx
        self.plane = 15        
        self.axis = 1
        
        self.xlabel = 'Axis X'
        self.ylabel = 'Axis Z' 
        
        self.voxels,self.label = self.load_voxel(self.idx)        
        self.colors = np.empty(self.voxels.shape, dtype=object)
        self.colors[:] = 'blue'
        self.colors[15][15][15][0] = 'red'
        
        self.DIM = self.voxels.shape[0]
        self.xx,self.yy = np.meshgrid(range(self.DIM+1),range(self.DIM+1))

        self.model = self.build_model()
        self.model
        
        self.plot()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(31,31,31,1)))
        model.add(Conv3D(filters=1,
                        kernel_size=3,
                        strides=1,
                        dilation_rate=3,
                        padding='same',
                        #activation='relu',
                        kernel_initializer=Constant(1.0/9),
                        #bias_initializer=Constant(1.0),
                        ))
        model.add(Conv3D(filters=1,
                        kernel_size=3,
                        dilation_rate=1,
                        padding='same',
                        kernel_initializer=Constant(1.0/9)))

        model.summary()
        return model
    
    def load_voxel(self,idx):
        voxels = self.hf['voxels'][idx]
        label = self.hf['labels'][idx]
        return voxels,label

    def next(self,event):
        self.idx += 1        
        self.voxels ,self.label = self.load_voxel(self.idx)        
        self.plot()
    

    def prev(self, event):
        self.idx -= 1
        self.voxels,self.label = self.load_voxel(self.idx)        
        self.plot()
        
    def show_conv_output(self,event):  
        output = self.model(np.expand_dims(self.voxels,axis=0))
        self.voxels = output[0]
        self.plot()

    def slider(self,val):
        self.plane = int(val)    
        self.plot()
        
    def plot(self):
        if self.axis == 0:
            image = self.voxels[self.plane,:,:,0]
            self.xlabel = 'Axis Y'
            self.ylabel = 'Axis Z'
        elif self.axis == 1:
            image = self.voxels[:,self.plane,:,0]
            self.xlabel = 'Axis X'
            self.ylabel = 'Axis Z'
        elif self.axis == 2:
            image = self.voxels[:,:,self.plane,0]
            self.xlabel = 'Axis X'
            self.ylabel = 'Axis Y'
        # Voxel
        ax1.clear()
        #ax1.voxels(self.voxels[...,0],facecolors=self.colors[...,0],edgecolor='k')        
        ax1.set_title('VOXEL IDX: {} LABEL: {}'.format(self.idx,self.label),fontsize=15)

        # Box
        DIM = self.DIM
        ax1.plot([0,DIM,DIM,0,0],[0,0,DIM,DIM,0],[0,0,0,0,0],'k')
        ax1.plot([0,DIM,DIM,0,0],[0,0,DIM,DIM,0],[DIM,DIM,DIM,DIM,DIM],'k')
        ax1.plot([0,DIM,DIM,0,0],[0,0,0,0,0],[0,0,DIM,DIM,0],'k')
        ax1.plot([0,DIM,DIM,0,0],[DIM,DIM,DIM,DIM,DIM],[0,0,DIM,DIM,0],'k')        
        
        # Imagem de corte
        ax2.clear()
        image = np.rot90(image,k=1) 
        img = ax2.imshow(image.max()-image,
                   origin='upper',
                   extent=(0,image.shape[0],0,image.shape[1]),
                   cmap='gray')
        img.set_clim(0,image.max())
        ax2.set_xlim([-2,33])
        ax2.set_ylim([-2,33])
        ax2.set_xlabel(self.xlabel,fontsize=12)
        ax2.set_ylabel(self.ylabel,fontsize=12)
        
        # Grid
        for i in range(self.xx.shape[0]):
            ax2.plot(self.xx[i,:],self.yy[i,:],color='k',linewidth=1)
        for j in range(self.xx.shape[1]):
            ax2.plot(self.xx[:,j],self.yy[:,j],color='k',linewidth=1)
            
            
callbacks = VoxelPlotCalbacks(trainset_file,idx=1000)

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

# Show convolution output
axconv = plt.axes([0.03, 0.02, 0.1, 0.05])
bconv = Button(axconv, 'Show Conv')
bconv.on_clicked(callbacks.show_conv_output)

#plt.subplots_adjust(bottom=0.1)
fig.tight_layout()

plt.show()