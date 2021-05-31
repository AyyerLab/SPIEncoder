import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RegularGridInterpolator as rgi
import random
from pylab import cross,dot,inv
import argparse
from VAE_network_MD import UnFlatten, VAE_LD, VAE_MD, VAE_HD
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy import interpolate
import os
import sys

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
#source activate /home/ayyerkar/.conda/envs/cuda
class Preprocessing:
    def __init__(self, fname, res, info):
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
        self.info = info
        self.planes_S_th = None
        self.rotation_sq_r_th = None
        self.select_number = None
        self.imagesize = None

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

        if (self.info == 1):
            rotation_sq0[:,4] = gain0
        if (self.info == 2):
            rotation_sq0[:,4] = labels0[:,1]

        self.labels = labels0[rlist]
        self.intens_input0 = intens_input00[rlist]
        self.gain = gain0[rlist]
        self.corr = corr0[rlist]
        self.rotation_sq = rotation_sq0[rlist]

    def scale_down(self):
        print('res = ', res)
        nl = self.intens_input0.shape[0]
        xo0 = np.arange(-121, 122, 1)
        yo0 = np.arange(-121, 122, 1)
        xo = xo0/121.
        yo = yo0/121.

        if (self.res == 0):
            xn0 = np.arange(-40, 41, 1)
            yn0 = np.arange(-40, 41, 1)
            xn = xn0/40.
            yn = yn0/40.
            self.intens_input_c = np.zeros([nl,81,81])
            for i in range(nl):
                input_map = self.intens_input0[i]
                f = interpolate.interp2d(xo, yo, input_map, kind='cubic')
                self.intens_input_c[i] = f(xn,yn)

        if (self.res == 1):
            xn0 = np.arange(-80, 81, 1)
            yn0 = np.arange(-80, 81, 1)
            xn = xn0/80.
            yn = yn0/80.
            self.intens_input_c = np.zeros([nl,161,161])
            for i in range(nl):
                input_map = self.intens_input0[i]
                f = interpolate.interp2d(xo, yo, input_map, kind='cubic')
                self.intens_input_c[i] = f(xn,yn)

        if (self.res == 2):
            self.intens_input_c = self.intens_input0  #HD

        self.imagesize = self.intens_input_c.shape[1]
        print('input image size = ', self.intens_input_c.shape)


    def shuffle(self, shuffle=False):
        size_half = np.int((self.imagesize - 1)/2)
        print('size_half = ', size_half)
        x0,y0 = np.indices((self.imagesize ,self.imagesize)); x0-=size_half; y0-=size_half
        intrad_2d = np.sqrt(x0**2+y0**2)
        if (res < 2):
            self.intens_input_c[:,intrad_2d > (self.imagesize - 1)//2] = 0.0
        if (res == 2):
            self.intens_input_c[:,intrad_2d > (self.imagesize - 1)//2] = 0.0
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
        xo0 = np.arange(-121, 122, 1)
        yo0 = np.arange(-121, 122, 1)
        xo = xo0/121.
        yo = yo0/121.
        f = interpolate.interp2d(xo, yo, input_plane, kind='cubic')
        if (self.res == 0):
            xn0 = np.arange(-40, 41, 1)
            yn0 = np.arange(-40, 41, 1)
            xn = xn0/40.
            yn = yn0/40.
            plane_c = f(xn, yn) #LD
        if (self.res == 1):
            xn0 = np.arange(-80, 81, 1)
            yn0 = np.arange(-80, 81, 1)
            xn = xn0/80.
            yn = yn0/80.
            plane_c = f(xn, yn) #MD
        if (self.res == 2):
            plane_c = input_plane #HD

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
        planes_S = np.zeros((n_Splanes,self.imagesize,self.imagesize,3))

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
        select_plane = planes_S_th[b_num*n_batches+i]
        volume_size = (recon_intens_3d_i.shape[2])
        #print('volume_size: ',recon_intens_3d_i.shape)
        #print('Plane_size: ',select_plane.shape)
        grid = select_plane.float()/(volume_size//2+1)
        recon_x_i = F.grid_sample(recon_intens_3d_i.view(1,1,volume_size,volume_size,volume_size), grid.view(1,volume_size,volume_size,1,3), mode='bilinear', padding_mode='zeros', align_corners=None)[0][0][:,:].reshape(volume_size,volume_size)
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

  intensize = intens.shape[2]
  print(epochs,' intens_size = ' ,intensize)
  for epoch in range(epochs):
    mu_all_0 = np.zeros((1,model.module.z_dim))
    logvar_all_0 = np.zeros((1,model.module.z_dim))
    for i in range(intens.shape[0]//n_batches):
        # Local batches and labels
        images = torch.from_numpy(intens[i*n_batches:(i+1)*n_batches]).view(n_batches,1,intensize,intensize)
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

  torch.save(model.module.state_dict(), fold_save+'/Vae_CNN3D_dict')
  mu_all = mu_all_0[1:,:]
  logvar_all = logvar_all_0[1:,:]
  return mu_all, logvar_all



def fit(intens, rotation_sq_r, model, optimizer, epochs, n_batches, planes_S_th):
  mu_all_0 = np.zeros((1,model.module.z_dim))
  logvar_all_0 = np.zeros((1,model.module.z_dim))
  recon_2D_all_0 = np.zeros((1,1,intens.shape[2],intens.shape[2]))

  for i in range(intens.shape[0]//n_batches):
      # Local batches and labels
      images = torch.from_numpy(intens[i*n_batches:(i+1)*n_batches]).view(n_batches,1,intens.shape[2],intens.shape[2])
      images = images.float().to(device)
      ori = torch.from_numpy(rotation_sq_r[i*n_batches:(i+1)*n_batches]).view(n_batches,5)
      ori = ori.float().to(device)
      recon_images, mu, logvar = model([images, ori])
      mu_all_0 = np.concatenate((mu_all_0, mu.detach().cpu().clone().numpy()),axis=0)
      logvar_all_0 = np.concatenate((logvar_all_0, logvar.detach().cpu().clone().numpy()),axis=0)
      loss, bce, bse, kld, recon_2D_x = loss_function(recon_images, planes_S_th, images, mu, logvar, i, n_batches)

      if (i < 3):
          recon_2D_all_0 = np.concatenate((recon_2D_all_0, recon_2D_x.detach().cpu().clone().numpy()),axis=0)

      if (i%n_batches == 0):
          to_print = "Batch[{}/{}] Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(i*n_batches,
                                  10000, loss.data.item(), bce.data.item(), bse.data.item(), kld.data.item())

          print(to_print)

  mu_all = mu_all_0[1:,:]
  logvar_all = logvar_all_0[1:,:]
  return mu_all, logvar_all, recon_2D_all_0


def main(fname, z_dim, info, res, epochs, batch_size, fold_save):
    preproc = Preprocessing(fname, res, info)
    preproc.load()
    preproc.scale_down()
    preproc.shuffle()
    preproc.slice_planes()
    try:
        os.mkdir(fold_save)
        print('create dir: ' + fold_save)
    except:
        print('cannot create dir')

    torch.manual_seed(0)
    if (res == 2):
        model = VAE_HD(device, z_dim=z_dim, info=info)
        if torch.cuda.device_count() > 1:
          print("Using VAE_HD on ", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(VAE_HD(device, z_dim=z_dim, info=info))

    if (res == 1):
        model = VAE_MD(device, z_dim=z_dim, info=info)
        if torch.cuda.device_count() > 1:
          print("Using VAE_MD on ", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(VAE_MD(device, z_dim=z_dim, info=info))

    if (res == 0):
        model = VAE_LD(device, z_dim=z_dim, info=info)
        if torch.cuda.device_count() > 1:
          print("Using VAE_LD on ", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(VAE_LD(device, z_dim=z_dim, info=info))

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print('Start training')
    mu_all0, logvar_all0 = train(preproc.intens, preproc.rotation_sq_r, model, optimizer, epochs, batch_size, preproc.planes_S_th, fold_save)
    print('Done!')


    #with h5py.File(fold_save+'/data_train.h5', "r") as data_tem:
    #    mu_all0 = data_tem['mu_all'][:]
    #    logvar_all0 = data_tem['logvar_all'][:]

    label_all = preproc.label_r[:mu_all0.shape[0]]
    with h5py.File(fold_save+'/data_train.h5', "w") as data_tem:
        data_tem['mu_all'] = mu_all0
        data_tem['logvar_all'] = logvar_all0
        data_tem['label_all'] = label_all


    print('Start fitting without eval')
    model.module.load_state_dict(torch.load(fold_save+'/Vae_CNN3D_dict'))
    #model.eval()
    mu_all1,logvar_all1 = fit(preproc.intens, preproc.rotation_sq_r, model, optimizer, epochs, batch_size, preproc.planes_S_th)
    print('Done!')

    with h5py.File(fold_save+'/data_no_eval.h5', "w") as data_tem:
        data_tem['mu_all'] = mu_all1
        data_tem['logvar_all'] = logvar_all1
        data_tem['label_all'] = label_all


    print('Start fitting with eval')
    model.module.load_state_dict(torch.load(fold_save+'/Vae_CNN3D_dict'))
    model.eval()
    mu_all2,logvar_all2, rec_sample = fit(preproc.intens, preproc.rotation_sq_r, model, optimizer, epochs, batch_size, preproc.planes_S_th)
    print('Done!')

    with h5py.File(fold_save+'/data_eval.h5', "w") as data_tem:
        data_tem['mu_all'] = mu_all2
        data_tem['logvar_all'] = logvar_all2
        data_tem['label_all'] = label_all

    if (z_dim == 2):
        plt.figure(figsize=(10,3))
        plt.subplot(131)
        plt.scatter(mu_all0[:,0], mu_all0[:,1], s=1, c=preproc.label_r[:mu_all0.shape[0],1])
        plt.xlabel('after training')
        plt.subplot(132)
        plt.scatter(mu_all1[:,0], mu_all1[:,1], s=1, c=preproc.label_r[:mu_all1.shape[0],1])
        plt.xlabel('fitting without eval')
        plt.subplot(133)
        plt.scatter(mu_all2[:,0], mu_all2[:,1], s=1, c=preproc.label_r[:mu_all2.shape[0],1])
        plt.xlabel('fitting with eval')
        plt.tight_layout()
        plt.savefig(fold_save+'/latent_space_gain.png')

    if (z_dim == 1):
        plt.figure(figsize=(10,3))
        plt.subplot(131)
        plt.scatter(mu_all0, label_all[:,1], s=1, c=label_all[:,1])
        plt.xlabel('after training')
        plt.subplot(132)
        plt.scatter(mu_all1, label_all[:,1], s=1, c=label_all[:,1])
        plt.xlabel('fitting without eval')
        plt.subplot(133)
        plt.scatter(mu_all2, label_all[:,1], s=1, c=label_all[:,1])
        plt.xlabel('fitting with eval')
        plt.tight_layout()
        plt.savefig(fold_save+'/latent_space_gain.png')



if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()
    config.read('config_test.ini')
    Path = str(config['DEFAULT']['PATH'])
    input_fname = str(config['DEFAULT']['input_fname'])
    z_dim = int(config['DEFAULT']['z_dim'])
    info = int(config['DEFAULT']['info'])
    batch_size = int(config['DEFAULT']['batch_size'])
    epoch = int(config['DEFAULT']['epoch'])
    res = int(config['DEFAULT']['resolution_option'])
    #res = 2

    print('res = ',res, 'z_dim = ',z_dim, 'epoch = ',epoch, 'batch_size = ',batch_size )
    if (res == 2):
        if (info == 0):
            save_foldername = 'HS_MODEL_z'+str(z_dim)+''
            print('info == 0, save folder path = ' + Path+save_foldername)
        if (info == 1):
            save_foldername = 'HS_MODEL_z'+str(z_dim)+'_gain'
            print('info == 1, save folder path = ' + Path+save_foldername)
        if (info == 2):
            save_foldername = 'HS_MODEL_z'+str(z_dim)+'_pca1'
            print('info == 2, save folder path = ' + Path+save_foldername)
        if (info > 2):
            print('info must between 0,1,2, invilad info! by default info == 1')
            info = 1
            save_foldername = 'HS_MODEL_z'+str(z_dim)+'_gain'

    if (res == 1):
        if (info == 0):
            save_foldername = 'MS_MODEL_z'+str(z_dim)+''
            print('info == 0, save folder path = ' + Path+save_foldername)
        if (info == 1):
            save_foldername = 'MS_MODEL_z'+str(z_dim)+'_gain'
            print('info == 1, save folder path = ' + Path+save_foldername)
        if (info == 2):
            save_foldername = 'MS_MODEL_z'+str(z_dim)+'_pca1'
            print('info == 2, save folder path = ' + Path+save_foldername)
        if (info > 2):
            print('info must between 0,1,2, invilad info! by default info == 1')
            info = 1
            save_foldername = 'MS_MODEL_z'+str(z_dim)+'_gain'

    if (res == 0):
        if (info == 0):
            save_foldername = 'LS_MODEL_z'+str(z_dim)+''
            print('info == 0, save folder path = ' + Path+save_foldername)
        if (info == 1):
            save_foldername = 'LS_MODEL_z'+str(z_dim)+'_gain'
            print('info == 1, save folder path = ' + Path+save_foldername)
        if (info == 2):
            save_foldername = 'LS_MODEL_z'+str(z_dim)+'_pca1'
            print('info == 2, save folder path = ' + Path+save_foldername)
        if (info > 2):
            print('info must between 0,1,2, invilad info! by default info == 1')
            info = 1
            save_foldername = 'LS_MODEL_z'+str(z_dim)+'_gain'


    epochs = 10
    test_name = '_X'
    fname = Path+input_fname
    fold_save = Path+save_foldername+test_name
    main(Path+input_fname,z_dim,info,res,epochs,batch_size,Path+save_foldername+test_name)
