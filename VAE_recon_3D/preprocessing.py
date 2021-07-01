import random

import numpy as np
import h5py
from scipy import interpolate
import torch

class Preprocessing:
    def __init__(self, fname, res, info, device):
        self.device = device
        self.fname = fname
        self.res = res
        self.info = info

        self._load()
        self._sample()
        self._slice_planes()

    def _load(self):
        # TODO: Fix hardcoded file path and selection criterion
        with h5py.File('/home/zhuangyu/VAE_cub42/ACC10k_PCA_coor.h5', 'r') as h5:
            labels0 = np.copy(h5['predict_PCA_coor'][:])
        rlist = np.where(((labels0[:,0]+0.15)*(-4) > labels0[:,1]) & (labels0[:,1] > -0.5))[0]
        self.select_number = len(rlist)

        with h5py.File(self.fname, 'r') as h5:
            intens_input00 = np.copy(h5['intens_input'][:])
            #self.ave_image = np.copy(h5['ave_image'][:])
            gain0 = np.copy(h5['gain'][:])
            corr0 = np.copy(h5['corr'][:])
            rotation_sq0 = np.copy(h5['quat_data'][:])
            self.qx1 = np.copy(h5['qx1'][:])
            self.qy1 = np.copy(h5['qy1'][:])
            self.qz1 = np.copy(h5['qz1'][:])

        if self.info == 1:
            rotation_sq0[:,4] = gain0

        self.labels = labels0[rlist]
        self.intens_input0 = intens_input00[rlist]
        self.gain = gain0[rlist]
        self.corr = corr0[rlist]
        self.rotation_sq = rotation_sq0[rlist]

    def _sample(self, shuffle=False):
        print('res = ', self.res)
        nl = self.intens_input0.shape[0]
        xo = np.arange(-121, 122, 1.) / 121.
        yo = np.arange(-121, 122, 1.) / 121.

        if self.res == 'LD':
            xn = np.arange(-40, 41, 1.) / 121
            yn = np.arange(-40, 41, 1.) / 121
            intens_input_c = np.zeros([nl, 81, 81])
            for i in range(nl):
                input_map = self.intens_input0[i]
                f = interpolate.interp2d(xo, yo, input_map, kind='cubic')
                intens_input_c[i] = f(xn,yn)
        elif self.res == 'MD':
            xn = np.arange(-80, 81, 1.) / 121
            yn = np.arange(-80, 81, 1.) / 121
            intens_input_c = np.zeros([nl,161,161])
            for i in range(nl):
                input_map = self.intens_input0[i]
                f = interpolate.interp2d(xo, yo, input_map, kind='cubic')
                intens_input_c[i] = f(xn,yn)
        elif self.res == 'HD':
            intens_input_c = self.intens_input0  #HD
        elif self.res == 'LD_Paper':
            intens_input_c = self.intens_input0[:,42:204:2,42:204:2]  #MD

        self.imagesize = intens_input_c.shape[1]
        print('input image size = ', intens_input_c.shape)

        size_half = np.int((self.imagesize - 1)/2)
        print('size_half = ', size_half)
        x0,y0 = np.indices((self.imagesize ,self.imagesize)); x0-=size_half; y0-=size_half
        intrad_2d = np.sqrt(x0**2+y0**2)
        if self.res < 2:
            intens_input_c[:,intrad_2d > (self.imagesize - 1)//2] = 0.0
        if self.res == 2:
            intens_input_c[:,intrad_2d > (self.imagesize - 1)//2] = 0.0

        if shuffle:
            randomList = random.sample(range(0, self.select_number), self.select_number)
            self.label_r = self.labels[randomList]
            self.rotation_sq_r = self.rotation_sq[randomList,:5]
            self.intens = intens_input_c/np.max(intens_input_c)*0.99
        else:
            print('shuffle = False')
            self.label_r = self.labels
            self.rotation_sq_r = self.rotation_sq
            self.intens = intens_input_c/np.max(intens_input_c)*0.99

    def scale_down_plane(self, input_plane):
        xo0 = np.arange(-121, 122, 1)
        yo0 = np.arange(-121, 122, 1)
        xo = xo0/121.
        yo = yo0/121.
        f = interpolate.interp2d(xo, yo, input_plane, kind='cubic')
        if self.res == 0:
            xn0 = np.arange(-40, 41, 1)
            yn0 = np.arange(-40, 41, 1)
            xn = xn0/121.
            yn = yn0/121.
            plane_c = f(xn, yn)/(len(xn)//2+1)  #LD
        if self.res == 1:
            xn0 = np.arange(-80, 81, 1)
            yn0 = np.arange(-80, 81, 1)
            xn = xn0/121.
            yn = yn0/121.
            plane_c = f(xn, yn)/(len(xn)//2+1)  #MD
        if self.res == 2:
            plane_c = input_plane/(input_plane.shape[1]//2+1)  #HD
        if self.res == 10:
            xn0 = np.arange(-40, 41, 1)
            yn0 = np.arange(-40, 41, 1)
            xn = xn0/60.
            yn = yn0/60.
            plane_c = f(xn, yn)/(len(xn)*2//2+1) #LD

        return plane_c

    def q_rotation(self, x, y, z, i):
        qw, qx, qy, qz, _ = self.rotation_sq_r[i]
        matrx = [[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                 [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                 [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]]

        t = np.transpose(np.array([x, y, z]), (1,2,0))
        x1,y1,z1 = np.transpose(t @ matrx, (2,0,1))
        return x1,y1,z1

    def _slice_planes(self):
        n_Splanes = self.rotation_sq_r.shape[0]
        planes_S = np.zeros((n_Splanes,self.imagesize,self.imagesize,3))

        x = self.scale_down_plane(self.qx1.reshape(243,243))
        y = self.scale_down_plane(self.qy1.reshape(243,243))
        z = self.scale_down_plane(self.qz1.reshape(243,243))
        for i in range(n_Splanes):
            x1,y1,z1 =  self.q_rotation(x,y,z, i)
            planes_S[i] = np.array([x1,y1,z1]).T

        self.planes_S_th = torch.from_numpy(planes_S).to(self.device)
        self.rotation_sq_r_th = torch.from_numpy(self.rotation_sq_r).to(self.device)
