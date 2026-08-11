import numpy as np
import matplotlib.pyplot as plt

class PaintPixel:
    def __init__(self,size=10):
        _,ax = plt.subplots()
                
        self.data = np.ones((size,size))
        self.image = ax.imshow(self.data,extent=[0,size,size,0],cmap='gray',
                                interpolation='none', vmin=0, vmax=1)            
        
        ax.set_xlim([0,10])
        ax.set_ylim([0,10])
        
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, size+1, 1))
        ax.set_yticks(np.arange(0, size+1, 1))
                            
        # Gridlines based on minor ticks
        ax.grid(color='k', linestyle='-', linewidth=1)
        
        self.press = False

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.image.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.image.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.image.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if event.inaxes != self.image.axes:
            return
        self.press = True        
        i,j = int(event.ydata), int(event.xdata)
        #print('coord: ',(i,j))   
        
        self.data[i,j] = 0
        self.image.set_data(self.data)
        
        self.image.figure.canvas.draw()
            

    def on_motion(self, event):
        if self.press is False:
            return
        i,j = int(event.ydata), int(event.xdata)
        #print('coord: ',(i,j))        

        self.data[i,j] = 0
        self.image.set_data(self.data)

        self.image.figure.canvas.draw()

    def on_release(self, event):
        """Clear button press information."""
        self.press = False
        self.image.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.image.figure.canvas.mpl_disconnect(self.cidpress)
        self.image.figure.canvas.mpl_disconnect(self.cidrelease)
        self.image.figure.canvas.mpl_disconnect(self.cidmotion)

dr = PaintPixel(31)
dr.connect()

plt.show()