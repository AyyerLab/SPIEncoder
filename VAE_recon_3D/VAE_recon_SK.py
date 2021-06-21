import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RegularGridInterpolator as rgi
import random
from pylab import cross,dot,inv
import argparse
from VAE_network_MD_SK import UnFlatten, VAE_LD, VAE_MD, VAE_HD
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy import interpolate
import os
import sys
from VAE_preprocessing import Preprocessing
from VAE_generating_volumes import generating

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
#source activate /home/ayyerkar/.conda/envs/cuda


def loss_function(recon_intens_3d, planes_S_th, x, mu, logvar, b_num, n_batches):
    def best_projection_layer(recon_intens_3d_i, x_i, i):
        select_plane = planes_S_th[b_num*n_batches+i]
        volume_size = (recon_intens_3d_i.shape[2])
        grid = select_plane.float()#/(volume_size*sampling//2+1)
        #print(volume_size, grid.max(), grid.min())
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
  return mu_all, logvar_all


def main(fname, z_dim, info, res, epochs, batch_size, fold_save, runtype):
    preproc = Preprocessing(fname, res, info, device)
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

    if (res == 10):
        model = VAE_LD(device, z_dim=z_dim, info=info)
        if torch.cuda.device_count() > 1:
          print("Using VAE_LD on ", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(VAE_LD(device, z_dim=z_dim, info=info))

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    fig = plt.figure(figsize=(8,5))
    for i in range(10):
        plt.subplot(4,5,i+1)
        plt.imshow(preproc.intens_input0[i*10])
    for i in range(10):
        plt.subplot(4,5,i+11)
        plt.imshow(preproc.intens_input_c[i*10])
    plt.tight_layout()
    plt.savefig(fold_save+"/Input_example.png")

    if (runtype == 'all'):
        print('Start training')
        mu_all0, logvar_all0 = train(preproc.intens, preproc.rotation_sq_r, model, optimizer, epochs, batch_size, preproc.planes_S_th, fold_save)
        print('Done!')


        label_all = preproc.label_r[:mu_all0.shape[0]]
        with h5py.File(fold_save+'/data_train.h5', "w") as data_tem:
            data_tem['mu_all'] = mu_all0
            data_tem['logvar_all'] = logvar_all0
            data_tem['label_all'] = label_all

        print('Start fitting without eval')
        model.module.load_state_dict(torch.load(fold_save+'/Vae_CNN3D_dict'))
        #model.eval()
        mu_all1, logvar_all1 = fit(preproc.intens, preproc.rotation_sq_r, model, optimizer, epochs, batch_size, preproc.planes_S_th)
        print('Done!')

        with h5py.File(fold_save+'/data_no_eval.h5', "w") as data_tem:
            data_tem['mu_all'] = mu_all1
            data_tem['logvar_all'] = logvar_all1
            data_tem['label_all'] = label_all


        print('Start fitting with eval')
        model.module.load_state_dict(torch.load(fold_save+'/Vae_CNN3D_dict'))
        model.eval()
        mu_all2, logvar_all2 = fit(preproc.intens, preproc.rotation_sq_r, model, optimizer, epochs, batch_size, preproc.planes_S_th)
        print('Done!')

        with h5py.File(fold_save+'/data_eval.h5', "w") as data_tem:
            data_tem['mu_all'] = mu_all2
            data_tem['logvar_all'] = logvar_all2
            data_tem['label_all'] = label_all

        generating(z_dim, info, res, fold_save, device)

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

    if (runtype == 'generate'):
        generating(z_dim, info, res, fold_save, device)



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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--runtype', default='all', help='running type: all (default); generate')
    args = parser.parse_args()
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
    fname = Path+input_fname
    print('Reading data from file: ',fname)
    fold_save = Path+save_foldername+test_name
    print('Saving data into folder: ',fold_save)
    runtype = args.runtype
    main(Path+input_fname,z_dim,info,res,epochs,batch_size,fold_save, runtype)
