import argparse
import random

import numpy as np
import h5py

import torch
from torch import optim
import torch.nn.functional as F

from model import VAE

class Preprocessing:
    def __init__(self, fname, res, device):
        self.fname = fname
        self.res = res
        self.device = device

        self.intens_input0 = None
        self.intens_input_c = None
        self.intens_input_m = None
        self.ave_image = None
        self.labels = None
        self.rotation_sq = None
        self.qx1 = None
        self.qy1 = None
        self.qz1 = None
        self.x = None
        self.y = None
        self.z = None
        self.intens_input_r = None
        self.label_r = None
        self.rotation_sq_r = None
        self.intens = None
        self.planes_S_th = None
        self.rotation_sq_r_th = None

    def load(self):
        with h5py.File(self.fname, 'r') as h5:
            self.intens_input0 = h5['intens_input'][:]
            self.ave_image = h5['ave_image'][:]
            self.labels = h5['labels'][:]
            self.rotation_sq = h5['rotation_sq'][:]
            self.qx1 = h5['qx1'][:]
            self.qy1 = h5['qy1'][:]
            self.qz1 = h5['qz1'][:]

    def scale_down(self):
        if self.res == 'low':
            self.intens_input_c = self.intens_input0[:,81:162,81:162] #LD
        elif self.res == 'med':
            self.intens_input_m = self.intens_input0[:,42:204,42:204]  #MD
            self.intens_input_c = self.intens_input_m[:,::2,::2]  #MD
        elif self.res == 'high':
            self.intens_input_c = self.intens_input0[:,::3,::3]  #HD
        else:
            raise ValueError('Unrecognized resolution setting %s' % self.res)

    def shuffle(self):
        ind = np.arange(81.) - 40
        x0, y0 = np.meshgrid(ind, ind, indices='ij')
        intrad_2d = np.sqrt(x0**2+y0**2)
        self.intens_input_c[:,intrad_2d > 40] = 0.0

        randomList = random.sample(range(0, 10000), 10000)
        self.intens_input_r = self.intens_input_c[randomList]
        self.label_r = self.labels[randomList]
        self.rotation_sq_r = self.rotation_sq[randomList]
        self.intens = self.intens_input_r/np.max(self.intens_input_r)*0.99

    def scale_down_plane(self, input_plane):
        if self.res == 0:
            plane_c = input_plane[81:162,81:162] #LD
        if self.res == 1:
            input_plane_1 = input_plane[42:204,42:204]  #MD
            plane_c = (input_plane_1[::2,::2])/2  #MD
        if self.res == 2:
            plane_c = input_plane[::3,::3]/3  #HD

        return plane_c

    def q_rotation(self, x, y, z, i):
        qw, qx, qy, qz, weight = self.rotation_sq_r[i]
        matrx = [[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                 [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                 [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]]

        t = np.transpose(np.array([x, y, z]), (1,2,0))
        x1,y1,z1 = np.transpose(t @ matrx, (2,0,1))
        return x1,y1,z1

    def slice_planes(self):
        n_Splanes = self.rotation_sq_r.shape[0]
        planes_S = np.zeros((n_Splanes,81,81,3))

        x = self.scale_down_plane(self.qx1.reshape(243,243))
        y = self.scale_down_plane(self.qy1.reshape(243,243))
        z = self.scale_down_plane(self.qz1.reshape(243,243))
        for i in range(n_Splanes):
            x1,y1,z1 =  self.q_rotation(x,y,z, i)
            planes_S[i] = np.array([x1,y1,z1]).T

        self.planes_S_th = torch.from_numpy(planes_S).to(self.device)
        self.rotation_sq_r_th = torch.from_numpy(self.rotation_sq_r).to(self.device)

class Trainer():
    def __init__(self, preproc, z_dim, device):
        self.preproc = preproc
        self.device = device

        self.model = VAE(device, z_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    @staticmethod
    def _to_numpy(arr):
        return arr.detach().cpu().clone().numpy()

    def loss_function(self, recon_intens_3d, x, mu, logvar, b_num, n_batches):
        def best_projection_layer(recon_intens_3d_i, x_i, i):
            select_plane = self.preproc.planes_S_th[b_num*20+i]
            grid = select_plane.float()/(41)
            recon_x_i = F.grid_sample(recon_intens_3d_i.view(1,1,81,81,81), grid.view(1,81,81,1,3), mode='bilinear', padding_mode='zeros', align_corners=None)[0][0][:,:].reshape(81,81)
            return recon_x_i

        recon_x = torch.zeros_like(x)
        for i in range(n_batches):
            recon_x_i = best_projection_layer(recon_intens_3d[i], x[i], i)
            recon_x[i] = recon_x_i

        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        BSE = torch.sum((recon_x - x)**2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD + BSE, BCE, BSE, KLD, recon_x

    def train(self, n_epochs, n_batches):
        for epoch in range(n_epochs):
            mu_all_0 = np.zeros((1,self.model.z_dim))
            logvar_all_0 = np.zeros((1,self.model.z_dim))

            for i in range(self.preproc.intens.shape[0]//n_batches):
                # Local batches and labels
                images = torch.from_numpy(self.preproc.intens[i*n_batches:(i+1)*n_batches]).view(n_batches,1,81,81)
                images = images.float().to(self.device)
                ori = torch.from_numpy(self.preproc.rotation_sq_r[i*n_batches:(i+1)*n_batches]).view(n_batches,5)
                ori = ori.float().to(self.device)
                recon_images, mu, logvar = self.model([images, ori])
                mu_all_0 = np.concatenate((mu_all_0, self._to_numpy(mu)), axis=0)
                logvar_all_0 = np.concatenate((logvar_all_0, self._to_numpy(logvar)), axis=0)
                loss, bce, bse, kld, recon_2D_x = self.loss_function(recon_images, images, mu, logvar, i, n_batches)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i%100 == 0:
                    to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(epoch+1,
                                            n_epochs, loss.data.item(), bce.data.item(), bse.data.item(), kld.data.item())
                    print(to_print)

        torch.save(self.model.state_dict(), 'Vae_cnn3D_dict')

    def fit(self, n_batches):
        self.model.load_state_dict(torch.load('Vae_cnn3D_dict'))
        mu_all_0 = np.zeros((1, self.model.z_dim))
        logvar_all_0 = np.zeros((1, self.model.z_dim))
        recon_2D_all_0 = np.zeros((1, 1, 81, 81))
        for i in range(500):
            # Local batches and labels
            images = torch.from_numpy(self.preproc.intens[i*n_batches:(i+1)*n_batches]).view(n_batches,1,81,81)
            images = images.float().to(self.device)
            ori = torch.from_numpy(self.preproc.rotation_sq_r[i*n_batches:(i+1)*n_batches]).view(n_batches,5)
            ori = ori.float().to(self.device)
            recon_images, mu, logvar = self.model([images, ori])
            mu_all_0 = np.concatenate((mu_all_0, self._to_numpy(mu)), axis=0)
            logvar_all_0 = np.concatenate((logvar_all_0, self._to_numpy(logvar)), axis=0)
            loss, bce, bse, kld, recon_2D_x = self.loss_function(recon_images, images, mu, logvar, i, n_batches)

            if i < 3:
                recon_2D_all_0 = np.concatenate((recon_2D_all_0, self._to_numpy(recon_2D_x)),axis=0)

            if i%20 == 0:
                to_print = "Batch[{}/{}] Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(i,
                                        500, loss.data.item(), bce.data.item(), bse.data.item(), kld.data.item())
                print(to_print)

        mu_all = mu_all_0[1:,:]
        logvar_all = logvar_all_0[1:,:]
        label_all = self.preproc.label_r[:10000]
        recon_2D_all = recon_2D_all_0[1:]

        rec_model_3D = np.zeros([5,81,81,81])
        latent_class = np.array([np.mean(mu_all[label_all == i], axis=0) for i in range(5)])
        latent_class_th = torch.from_numpy(latent_class).to(self.device).float()
        for i in range(5):
            rec_model_3D[i] = self._to_numpy(self.model.decode(latent_class_th[i]))

        with h5py.File('Figure_t4/temporal_data.h5','w') as data_tem:
            data_tem['mu_all'] = mu_all
            data_tem['logvar_all'] = logvar_all
            #data_tem['omu_all'] = omu_all
            #data_tem['ologvar_all'] = ologvar_all
            data_tem['label_all'] = label_all
            data_tem['recon_2D_all'] = recon_2D_all
            data_tem['rec_model_3D'] = rec_model_3D

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fname', default='', help='Path to input data H5 file')
    parser.add_argument('-z', '--z_dim', type=int, default=20, help='Latent space dimensions')
    parser.add_argument('-b', '--batch', type=int, default=20, help='Batch size')
    parser.add_argument('-r', '--resolution', default='low', help='Resolution string (low, med, high)')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-d', '--device_num', type=int, default=0, help='Device index for multi-GPU machines')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda:%d'%args.device_num if torch.cuda.is_available() else 'cpu')

    preproc = Preprocessing(args.fname, args.resolution, device)
    preproc.load()
    preproc.scale_down()
    preproc.shuffle()
    preproc.slice_planes()

    trainer = Trainer(preproc, args.z_dim, device)
    print('Start training')
    trainer.train(args.epochs, args.batch)
    print('Done!')

if __name__ == '__main__':
    main()
