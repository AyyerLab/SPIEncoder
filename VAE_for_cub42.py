import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RegularGridInterpolator as rgi
import random
from pylab import cross,dot,inv
import argparse
from VAE_network import UnFlatten, VAE
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os

# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


#source activate /home/ayyerkar/.conda/envs/cuda
class Preprocessing:
    def __init__(self, fname, res):
        self.intens_input0 = None
        self.intens_input_c = None
        self.intens_input_m = None
        self.ave_image = None
        self.labels0 = None
        self.labels = None
        self.gain = None
        self.corr = None
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
        self.select_number = None

    def load(self):
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

        rotation_sq0[:,4] = labels0[:,1]
        self.labels = labels0[rlist]
        self.intens_input0 = intens_input00[rlist]
        self.gain = gain0[rlist]
        self.corr = corr0[rlist]
        self.rotation_sq = rotation_sq0[rlist]

    def scale_down(self):
        if self.res == 0:
            self.intens_input_c = self.intens_input0[:,81:162,81:162] #LD
        if self.res == 1:
            self.intens_input_m = self.intens_input0[:,42:204,42:204]  #MD
            self.intens_input_c = self.intens_input_m[:,::2,::2]  #MD
        if self.res == 2:
            self.intens_input_c = self.intens_input0[:,::3,::3]  #HD


    def shuffle(self, shuffle=False):

        x0,y0 = np.indices((81,81)); x0-=40; y0-=40
        intrad_2d = np.sqrt(x0**2+y0**2)
        self.intens_input_c[:,intrad_2d > 40] = 0.0
        if shuffle == True:
            randomList = random.sample(range(0, self.select_number), self.select_number)
            self.intens_input_r = self.intens_input_c[randomList]
            self.label_r = self.labels[randomList]
            self.rotation_sq_r = self.rotation_sq[randomList,:5]
            self.intens = self.intens_input_r/np.max(self.intens_input_r)*0.99
        if shuffle == False:
            print('shuffle = False')
            self.intens_input_r = self.intens_input_c
            self.label_r = self.labels
            self.rotation_sq_r = self.rotation_sq
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
        arrth = recon_intens_3d[i]
        symarrth = 0.25 * (arrth + torch.flip(arrth, [1]) + torch.flip(arrth, [2]) + torch.flip(arrth, [3]))
        symarrth = (symarrth + symarrth.transpose(1,3).transpose(2,3) + symarrth.transpose(1,2).transpose(2,3)) / 3.
        symarrth = 0.5 * (symarrth + torch.flip(symarrth, [1, 2, 3]))
        recon_intens_3d_sym = symarrth
        recon_x_i = best_projection_layer(recon_intens_3d_sym, x[i], i)
        recon_x[i] = recon_x_i

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    BSE = torch.sum((recon_x - x)**2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD + BSE, BCE, BSE, KLD, recon_x


def train(intens, rotation_sq_r, model, optimizer, epochs, n_batches, planes_S_th, fold_save):
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

    torch.save(model.state_dict(), fold_save+'/Vae_CNN3D_dict')



def fit(intens, rotation_sq_r, model, optimizer, epochs, n_batches, planes_S_th):
  mu_all_0 = np.zeros((1,model.z_dim))
  logvar_all_0 = np.zeros((1,model.z_dim))
  recon_2D_all_0 = np.zeros((1,1,81,81))

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

      if (i < 3):
          recon_2D_all_0 = np.concatenate((recon_2D_all_0, recon_2D_x.detach().cpu().clone().numpy()),axis=0)

      if (i%20 == 0):
          to_print = "Batch[{}/{}] Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(i,
                                  500, loss.data.item(), bce.data.item(), bse.data.item(), kld.data.item())
          print(to_print)

  mu_all = mu_all_0[1:,:]
  logvar_all = logvar_all_0[1:,:]
  return mu_all, logvar_all


def main(fname, z_dim, info_dim, res, epochs, batch_size, fold_save):
#main(Path+input_fname,z_dim,info_dim,res,epoch,batch_size,Path+save_foldername)
    res = 1
    batch_size = 20
    fname = Path+input_fname
    epochs = epoch
    fold_save = Path+save_foldername
    #
    preproc = Preprocessing(fname, res)
    preproc.load()
    preproc.scale_down()
    preproc.shuffle()
    preproc.slice_planes()
    try:
        os.mkdir(fold_save)
    except:
        print('cannot creat dir')

    model = VAE(z_dim=z_dim, info_dim=info_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print('Start training')
    train(preproc.intens, preproc.rotation_sq_r, model, optimizer, epochs, batch_size, preproc.planes_S_th, fold_save)
    print('Done!')

    print('Start fitting')
    model.load_state_dict(torch.load(fold_save+'/Vae_CNN3D_dict'))
    model.eval()
    mu_all,logvar_all = fit(preproc.intens, preproc.rotation_sq_r, model, optimizer, epochs, batch_size, preproc.planes_S_th)
    print('Done!')
    plt.figure(figsize=(8,6))
    plt.scatter(mu_all[:,0], mu_all[:,1], s=1, c=preproc.label_r[:mu_all.shape[0],1])
    plt.tight_layout()
    plt.colorbar(label='PCA-1')
    plt.savefig(fold_save+'/latent_space.png')



    with h5py.File(fold_save+'/data.h5', "w") as data_tem:
        data_tem['mu_all'] = mu_all
        data_tem['logvar_all'] = logvar_all
        data_tem['label_all'] = preproc.label_r[:mu_all.shape[0]]



if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()
    config.read('config_test.ini')
    Path = str(config['DEFAULT']['PATH'])
    input_fname = str(config['DEFAULT']['input_fname'])
    z_dim = int(config['DEFAULT']['z_dim'])
    info_dim = int(config['DEFAULT']['info_dim'])
    batch_size = int(config['DEFAULT']['batch_size'])
    epoch = int(config['DEFAULT']['epoch'])
    res = int(config['DEFAULT']['resolution_option'])
    save_foldername = 'MODEL_z_'+str(z_dim)+'_info_'+str(info_dim)
    main(Path+input_fname,z_dim,info_dim,res,epoch,batch_size,Path+save_foldername)
