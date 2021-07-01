import sys

import numpy as np
import h5py
import pandas as pd
import pylab as P
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import colors
from scipy.interpolate import RegularGridInterpolator as rgi
import random
from pylab import cross,dot,inv
import os

import numpy as np
import torch
from torch import optim
from network import UnFlatten, VAE_LD, VAE_MD, VAE_HD

# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

class TrainedVAE():

    def __init__(self, state_dict_file, res, device, z_dim=2, info=1):
        self.res = res
        if (self.res == 0):
            self.intens_size = 81
        if (self.res == 1):
            self.intens_size = 161
        if (self.res == 2):
            self.intens_size = 243


        self.device = device
        torch.manual_seed(0)
        self.z_dim = z_dim
        self.info = info
        self._load_model(state_dict_file)
        self._gen_ave_3d()

    def _load_model(self, state_dict):
        if (self.res == 0):
            self.model = VAE_LD(self.device, z_dim=self.z_dim, info=self.info).to(self.device)
        if (self.res == 1):
            self.model = VAE_MD(self.device, z_dim=self.z_dim, info=self.info).to(self.device)
        if (self.res == 2):
            self.model = VAE_HD(self.device, z_dim=self.z_dim, info=self.info).to(self.device)

        #self.model = VAE_MD(self.device, z_dim=self.z_dim, info=self.info).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.load_state_dict(torch.load(state_dict, map_location=self.device))
        self.model.eval()

    def _gen_ave_3d(self):
        ind = np.arange(self.intens_size) - self.intens_size // 2
        x, y, z = np.meshgrid(ind, ind, ind, indexing='ij')
        rad = np.sqrt(x*x + y*y + z*z)
        self.ave_3d = 500 * rad**-3.3 + 0.0005
        self.ave_3d[rad<5] = 0.

    def decode(self, coords):
        tcoords = torch.from_numpy(np.array(coords)).to(self.device).float()
        intens = self.model.decode(tcoords).detach().cpu().numpy()[0,0]

        symintens = symintens = 0.25 * (intens + intens[::-1] + intens[:,::-1] + intens[:,:,::-1])
        symintens = (symintens + symintens.transpose(1,2,0) + symintens.transpose(2,0,1)) / 3.
        symintens = 0.5 * (symintens + symintens[::-1,::-1,::-1])

        return symintens * self.ave_3d

class VAEExplorer(QtWidgets.QMainWindow):
    def __init__(self, fold_save, z_dim, res):
        super().__init__()
        self._init_ui()
        self.z_dim = z_dim
        self.res = res
        if (self.res == 0):
            self.intens_size = 81
        if (self.res == 1):
            self.intens_size = 161
        if (self.res == 2):
            self.intens_size = 243

    def _init_ui(self):
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout()
        widget.setLayout(layout)
        self.path = fold_save

        panel = QtWidgets.QVBoxLayout()
        layout.addLayout(panel)
        self.scatter_fig = P.figure(figsize=(6, 4))
        self.scatter_fig.add_subplot(111)
        self.scatter_canvas = FigureCanvas(self.scatter_fig)
        self.scatter_canvas.mpl_connect('button_press_event', self._scatter_clicked)
        navbar = NavigationToolbar(self.scatter_canvas, self)
        panel.addWidget(navbar)
        panel.addWidget(self.scatter_canvas)

        line = QtWidgets.QHBoxLayout()
        panel.addLayout(line)
        label = QtWidgets.QLabel('Latent space file:')
        line.addWidget(label)
        self.scatter_fname = QtWidgets.QLineEdit(self.path + '/data_eval.h5')
        line.addWidget(self.scatter_fname)
        button = QtWidgets.QPushButton('Load')
        button.clicked.connect(self._load_scatter_df)
        line.addWidget(button)
        self._load_scatter_df()
        label = QtWidgets.QLabel('Color key:')
        line.addWidget(label)
        self.color_key = QtWidgets.QComboBox()
        self.color_key.addItems(self.scatter_df.keys())
        self.color_key.setCurrentIndex(2)
        self.color_key.currentIndexChanged.connect(self._plot_scatter)
        line.addWidget(self.color_key)
        line.addStretch(1)

        panel = QtWidgets.QVBoxLayout()
        layout.addLayout(panel)
        self.model_fig = P.figure(figsize=(8, 3))
        self.model_fig.add_subplot(131)
        self.model_fig.add_subplot(132)
        self.model_fig.add_subplot(133)
        self.model_canvas = FigureCanvas(self.model_fig)
        navbar = NavigationToolbar(self.model_canvas, self)
        panel.addWidget(navbar)
        panel.addWidget(self.model_canvas)

        line = QtWidgets.QHBoxLayout()
        panel.addLayout(line)
        label = QtWidgets.QLabel('Latent space file:')
        line.addWidget(label)
        self.vae_fname = QtWidgets.QLineEdit(self.path + '/Vae_CNN3D_dict')
        line.addWidget(self.vae_fname)
        button = QtWidgets.QPushButton('Load')
        button.clicked.connect(self._load_vae)
        line.addWidget(button)
        self._load_vae()
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        panel.addLayout(line)
        label = QtWidgets.QLabel('z_0')
        line.addWidget(label)
        self.z0 = QtWidgets.QLineEdit('0.0')
        self.z0.editingFinished.connect(self._draw_model)
        line.addWidget(self.z0)
        label = QtWidgets.QLabel('z_1')
        line.addWidget(label)
        self.z1 = QtWidgets.QLineEdit('0.0')
        self.z1.editingFinished.connect(self._draw_model)
        line.addWidget(self.z1)
        line.addStretch(1)

        self._plot_scatter()
        self.show()

    def _load_scatter_df(self):
        fname = self.scatter_fname.text()
        with h5py.File(fname, 'r') as f:
            mu = f['mu_all'][:]
            label = f['label_all'][:]
            logvar = f['logvar_all'][:]

        self.scatter_df = pd.DataFrame({'x':mu[:,0],
            'y':mu[:,1],
            'PCA-0':label[:,0],
            'PCA-1':label[:,1],
            'PCA-2':label[:,2],
            'LogVar-0':logvar[:,0],
            'LogVar-1':logvar[:,1],
            'Err-0':np.exp(logvar[:,0]/2),
            'Err-1':np.exp(logvar[:,1]/2)})

    def _load_vae(self):
        fname = self.vae_fname.text()
        self.vae = TrainedVAE(fname, res, device)

    def _plot_scatter(self, value=None):
        ax = self.scatter_fig.axes[0]
        ax.clear()
        ckey = self.color_key.currentText()

        df = self.scatter_df
        ax.scatter(df['x'], df['y'], c=df[ckey])
        self.scatter_canvas.draw()

    def _scatter_clicked(self, event):
        if event.inaxes is None or event.button != matplotlib.backend_bases.MouseButton.LEFT:
            return
        self.z0.setText('%.2f' % event.xdata)
        self.z1.setText('%.2f' % event.ydata)
        ax = self.scatter_fig.axes[0]
        for line in ax.lines:
            line.remove()
        ax.plot(event.xdata, event.ydata, c='red', marker='x')
        self.scatter_canvas.draw()
        self._draw_model(z0=event.xdata, z1=event.ydata)

    def rotation(self, x,y,z,axis,angle):
        a = angle
        mx = [[1, 0, 0],[0, np.cos(a), -np.sin(a)],[0, np.sin(a), np.cos(a)]]
        my = [[np.cos(a), 0, np.sin(a)],[0, 1, 0],[-np.sin(a), 0, np.cos(a)]]
        mz = [[np.cos(a), -np.sin(a), 0],[np.sin(a), np.cos(a), 0],[0, 0, 1]]
        Matrix_r = [mx,my,mz]
        M_axis = Matrix_r[axis]
        t = np.transpose(np.array([x,y,z]), (1,2,0))
        x1,y1,z1 = np.transpose(t @ M_axis, (2,0,1))
        return x1,y1,z1

    def _draw_model(self, z0=None, z1=None):
        if z0 is None:
            z0 = float(self.z0.text())
        if z1 is None:
            z1 = float(self.z1.text())
        Volume = self.vae.decode([z0, z1])
        scale = self.intens_size
        half_scale = scale//2
        x0,y0,z0 = np.indices((scale,scale,scale))
        x0-=half_scale; y0-=half_scale; z0-=half_scale
        Interpolating_model_m = rgi((x0[:,0,0],y0[0,:,0],z0[0,0,:]), Volume, bounds_error=False, fill_value=0.0)

        U = np.array([0, 0, 1])
        V011 = np.array([0, 1, 1])
        V111 = np.array([1, 1, 1])
        x001,y001 = np.indices((scale,scale)); x001-=half_scale; y001-=half_scale
        z001 = x001*0.0
        x011,y011,z011 = self.rotation(x001,y001,z001,0,np.pi/4)
        x111_0,y111_0,z111_0 = self.rotation(x001,y001,z001,0,0.955316618)
        x111,y111,z111 = self.rotation(x111_0,y111_0,z111_0,2,np.pi/4)

        image_001 = Interpolating_model_m(np.array([x001,y001,z001]).T).ravel()
        image_011 = Interpolating_model_m(np.array([x011,y011,z011]).T).ravel()
        image_111 = Interpolating_model_m(np.array([x111,y111,z111]).T).ravel()
        intens_100 = image_001.reshape(scale,scale)
        intens_110 = image_011.reshape(scale,scale)
        intens_111 = image_111.reshape(scale,scale)

        ax = self.model_fig.axes[0]
        ax.clear()
        ax.imshow(intens_100, norm=colors.LogNorm(vmin=1e-4))
        ax = self.model_fig.axes[1]
        ax.clear()
        ax.imshow(intens_110, norm=colors.LogNorm(vmin=1e-4))
        ax = self.model_fig.axes[2]
        ax.clear()
        ax.imshow(intens_111, norm=colors.LogNorm(vmin=1e-4))
        self.model_canvas.draw()

if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()
    config.read('config_real.ini')
    Path = str(config['DEFAULT']['PATH'])
    input_fname = str(config['DEFAULT']['input_fname'])
    z_dim = int(config['DEFAULT']['z_dim'])
    info = int(config['DEFAULT']['info'])
    batch_size = int(config['DEFAULT']['batch_size'])
    epochs = int(config['DEFAULT']['epochs'])
    res = int(config['DEFAULT']['resolution_option'])
    test_name = str(config['DEFAULT']['test_name'])
    sample_xp = str(config['DEFAULT']['sample_xp'])
    sample_x = sample_xp.split(',')
    sample_x = [float(i) for i in sample_x]

    #res = 2

    print('res = ',res, 'z_dim = ',z_dim, 'epochs = ',epochs, 'batch_size = ',batch_size )
    if (res == 2):
        if (info == 0):
            save_foldername = 'HS_MODEL_z'+str(z_dim)+''
            print('info = 0, save folder path = ' + Path+save_foldername)
        if (info > 0):
            info = 1
            save_foldername = 'HS_MODEL_z'+str(z_dim)+'_G'
            print('info = 1, save folder path = ' + Path+save_foldername)

    if (res == 1):
        if (info == 0):
            save_foldername = 'MS_MODEL_z'+str(z_dim)+''
            print('info = 0, save folder path = ' + Path+save_foldername)
        if (info > 0):
            info = 1
            save_foldername = 'MS_MODEL_z'+str(z_dim)+'_G'
            print('info = 1, save folder path = ' + Path+save_foldername)

    if (res == 0):
        if (info == 0):
            save_foldername = 'LS_MODEL_z'+str(z_dim)+''
            print('info = 0, save folder path = ' + Path+save_foldername)
        if (info > 0):
            info = 1
            save_foldername = 'LS_MODEL_z'+str(z_dim)+'_G'
            print('info = 1, save folder path = ' + Path+save_foldername)

    if (res == 10):
        if (info == 0):
            save_foldername = 'MLS_MODEL_z'+str(z_dim)+''
            print('info = 0, save folder path = ' + Path+save_foldername)
        if (info > 0):
            info = 1
            save_foldername = 'MLS_MODEL_z'+str(z_dim)+'_G'
            print('info = 1, save folder path = ' + Path+save_foldername)

    test_name = '_SK'
    fold_save = Path+save_foldername+test_name
    print('Reading data from folder: ',fold_save)
    app = QtWidgets.QApplication(sys.argv)
    v = VAEExplorer(fold_save, z_dim, res)
    sys.exit(app.exec_())
