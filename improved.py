import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RegularGridInterpolator as rgi
import random
from pylab import cross,dot,inv
import argparse
from model import UnFlatten, VAE
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


#source activate /home/ayyerkar/.conda/envs/cuda
class Preprocessing:
    def __init__(self, fname, res):
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
        self.fname = fname
        self.intens_input_r = None
        self.label_r = None
        self.rotation_sq_r = None
        self.intens = None
        self.res = res
        self.planes_S_th = None
        self.rotation_sq_r_th = None

    def load(self):
        with h5py.File(self.fname, 'r') as h5:
            self.intens_input0 = np.copy(h5['intens_input'][:])
            self.ave_image = np.copy(h5['ave_image'][:])
            self.labels = np.copy(h5['labels'][:])
            self.rotation_sq = np.copy(h5['rotation_sq'][:])
            self.qx1 = np.copy(h5['qx1'][:])
            self.qy1 = np.copy(h5['qy1'][:])
            self.qz1 = np.copy(h5['qz1'][:])

    def scale_down(self):
        if self.res == 0:
            self.intens_input_c = self.intens_input0[:,81:162,81:162] #LD
        if self.res == 1:
            self.intens_input_m = self.intens_input0[:,42:204,42:204]  #MD
            self.intens_input_c = self.intens_input_m[:,::2,::2]  #MD
        if self.res == 2:
            self.intens_input_c = self.intens_input0[:,::3,::3]  #HD


    def shuffle(self):
        x0,y0 = np.indices((81,81)); x0-=40; y0-=40
        intrad_2d = np.sqrt(x0**2+y0**2)
        self.intens_input_c[:,intrad_2d > 40] = 0.0

        randomList = random.sample(range(0, 10000), 10000)
        self.intens_input_r = self.intens_input_c[randomList]
        self.label_r = self.labels[randomList]
        self.rotation_sq_r = self.rotation_sq[randomList]
        self.intens = self.intens_input_r/np.max(self.intens_input_r)*0.99

    def scale_down_plane(self, input_plane):
        if (self.res == 0):
            plane_c = input_plane[81:162,81:162] #LD
        if (self.res == 1):
            input_plane_1 = input_plane[42:204,42:204]  #MD
            plane_c = (input_plane_1[::2,::2])/2  #MD
        if (self.res == 2):
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

        self.planes_S_th = torch.from_numpy(planes_S).to(device)
        self.rotation_sq_r_th = torch.from_numpy(self.rotation_sq_r).to(device)


def loss_function(recon_intens_3d, planes_S_th, x, mu, logvar, b_num, n_batches):
    def best_projection_layer(recon_intens_3d_i, x_i, i):
        select_plane = planes_S_th[b_num*20+i]
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


def train(intens, rotation_sq_r, model, optimizer, epochs, n_batches, planes_S_th):
  for epoch in range(epochs):
    mu_all_0 = np.zeros((1,model.z_dim))
    logvar_all_0 = np.zeros((1,model.z_dim))
    for i in range(intens.shape[0]//n_batches):
        # Local batches and labels
        images = torch.from_numpy(intens[i*n_batches:(i+1)*n_batches]).view(n_batches,1,81,81)
        images = images.float().to(device)
        ori = torch.from_numpy(rotation_sq_r[i*n_batches:(i+1)*n_batches]).view(n_batches,5)
        ori = ori.float().to(device)
        recon_images, mu, logvar = model([images, ori])
        mu_all_0 = np.concatenate((mu_all_0, mu.detach().cpu().clone().numpy()),axis=0)
        logvar_all_0 = np.concatenate((logvar_all_0, logvar.detach().cpu().clone().numpy()),axis=0)
        loss, bce, bse, kld, recon_2D_x = loss_function(recon_images, planes_S_th, images, mu, logvar, i, n_batches)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i%100 == 0):
            to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(epoch+1,
                                    epochs, loss.data.item(), bce.data.item(), bse.data.item(), kld.data.item())
            print(to_print)

  torch.save(model.state_dict(), 'Vae_cnn3D_dict')


def fit(intens, rotation_sq_r, model, optimizer, epochs, n_batches, planes_S_th):
  model.load_state_dict(torch.load('Vae_cnn3D_dict'))
  mu_all_0 = np.zeros((1,model.z_dim))
  logvar_all_0 = np.zeros((1,model.z_dim))
  recon_2D_all_0 = np.zeros((1,1,81,81))
  for i in range(500):
      # Local batches and labels
      images = torch.from_numpy(intens[i*n_batches:(i+1)*n_batches]).view(n_batches,1,81,81)
      images = images.float().to(device)
      recon_images, mu, logvar = model([images, ori])
      mu_all_0 = np.concatenate((mu_all_0, mu.detach().cpu().clone().numpy()),axis=0)
      logvar_all_0 = np.concatenate((logvar_all_0, logvar.detach().cpu().clone().numpy()),axis=0)
      loss, bce, bse, kld, recon_2D_x = loss_function(recon_images, images, mu, logvar, i)

      if (i < 3):
          recon_2D_all_0 = np.concatenate((recon_2D_all_0, recon_2D_x.detach().cpu().clone().numpy()),axis=0)

      if (i%20 == 0):
          to_print = "Batch[{}/{}] Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(i,
                                  500, loss.data.item(), bce.data.item(), bse.data.item(), kld.data.item())
          print(to_print)

  mu_all = mu_all_0[1:,:]
  logvar_all = logvar_all_0[1:,:]
  label_all = label_r[:10000]
  recon_2D_all = recon_2D_all_0[1:]

def main(fname, res, epochs, batch_size):
    preproc = Preprocessing(fname, res)
    preproc.load()
    preproc.scale_down()
    preproc.shuffle()
    preproc.slice_planes()

    model = VAE(z_dim=z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print('Start training')
    train(preproc.intens, preproc.rotation_sq_r, model, optimizer, epochs, batch_size, preproc.planes_S_th)
    print('Done!')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fname', default='', help='You have to parse the file name!')
    parser.add_argument('-z', '--z_dim', type=int, default=20, help='z dim')
    parser.add_argument('-b', '--batch', type=int, default=20, help='batch size')
    parser.add_argument('-r', '--resolution', type=int, default=1, help='resolution')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='epochs')
    args = parser.parse_args()
    main(args.fname, args.resolution, args.epochs, args.batch)
