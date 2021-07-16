import random

import numpy as np
import h5py
import torch

class Preprocessing:
    def __init__(self, input_path, model, info, device):
        self.device = device
        self.input_path = input_path
        self.model = model
        self.info = info

        self.load()
        self.sample()
        self.slice_planes()

    def load(self):
        # TODO: Fix hardcoded file path and selection criterion
        with h5py.File(self.input_path + 'ACC10k_PCA_coor_new_labels.h5', 'r') as h5f:
            labels0 = h5f['predict_PCA_coor'][:]
        rlist = np.where(((labels0[:,0]+0.15)*(-4) > labels0[:,1]) & (labels0[:,1] > -0.5))[0]
        self.select_number = len(rlist)

        with h5py.File(self.input_path + 'VAE_input_pattern.h5', 'r') as h5f:
            intens_input00 = h5f['intens_input'][:]
            #self.ave_image = h5f['ave_image'][:]
            gain0 = h5f['gain'][:]
            corr0 = h5f['corr'][:]
            rotation_sq0 = h5f['quat_data'][:]
            self.qx1 = h5f['qx1'][:]
            self.qy1 = h5f['qy1'][:]
            self.qz1 = h5f['qz1'][:]

        if self.info == 5:
            rotation_sq0[:,4] = gain0

        self.labels = labels0[rlist]
        self.intens_input0 = intens_input00[rlist]
        self.gain = gain0[rlist]
        self.corr = corr0[rlist]
        self.rotation_sq = rotation_sq0[rlist]

    def sample(self, shuffle=False):
        print('model type = ', type(self.model).__name__)
        intens_input_c = self.model.module.preproc_sample_intens(self.intens_input0)

        if shuffle:
            random_order = random.sample(range(0, self.select_number), self.select_number)
            self.label_r = self.labels[random_order]
            self.rotation_sq_r = self.rotation_sq[random_order,:5]
            self.intens = intens_input_c / np.max(intens_input_c) * 0.99
        else:
            print('shuffle = False')
            self.label_r = self.labels
            self.rotation_sq_r = self.rotation_sq
            self.intens = intens_input_c / np.max(intens_input_c) * 0.99

    def _q_rotation(self, x, y, z, i):
        qw, qx, qy, qz, _ = self.rotation_sq_r[i]
        matrx = [[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                 [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                 [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]]

        return np.transpose(np.transpose(np.array([x, y, z]), (1,2,0)) @ matrx, (2,0,1))

    def slice_planes(self):
        imagesize = self.intens.shape[1]
        n_splanes = self.rotation_sq_r.shape[0]
        planes_s = np.zeros((n_splanes,imagesize,imagesize,3))

        x = self.model.module.preproc_scale_down_plane(self.qx1.reshape(243, 243))
        y = self.model.module.preproc_scale_down_plane(self.qy1.reshape(243, 243))
        z = self.model.module.preproc_scale_down_plane(self.qz1.reshape(243, 243))
        for i in range(n_splanes):
            planes_s[i] = self._q_rotation(x, y, z, i).T

        self.planes_s_th = torch.from_numpy(planes_s).to(self.device)
        self.rotation_sq_r_th = torch.from_numpy(self.rotation_sq_r).to(self.device)
