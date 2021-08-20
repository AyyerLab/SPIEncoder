import os
import configparser
import argparse

import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional

import networks
from preprocessing import Preprocessing
from generating_volumes import generating

class VAEReconstructor():
    def __init__(self, config_fname, device, learning_rate=1.e-3):
        self.device = device

        self._parse_config(config_fname)
        self._get_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.preproc = Preprocessing(self.input_path, self.model, self.n_info, self.device)

    def _parse_config(self, config_fname):
        config = configparser.ConfigParser()
        config.read(config_fname)

        # Get reconstruction parameters
        self.input_path = config.get('reconstructor', 'input_path')
        self.z_dim = config.getint('reconstructor', 'z_dim', fallback=1)
        self.n_info = config.getint('reconstructor', 'n_info')
        self.batch_size = config.getint('reconstructor', 'batch_size')
        self.n_epochs = config.getint('reconstructor', 'n_epochs')
        self.res_str = config.get('reconstructor', 'resolution_string')

        print('resolution_type =', self.res_str)
        print(self.z_dim, 'dimensional latent space')
        print(self.n_epochs, 'epochs with a batch size of', self.batch_size)

        # Get output folder information
        overwrite = config.getboolean('reconstructor', 'overwrite_output', fallback=False)
        parent_dir = config.get('reconstructor', 'output_parent', fallback='data/')
        suffix = config.get('reconstructor', 'output_suffix', fallback='')
        self.output_folder = parent_dir + '/%s_%d_%s/' % (self.res_str, self.z_dim, suffix)
        os.makedirs(self.output_folder, exist_ok=overwrite)

        print('Saving data into folder: ', self.output_folder)

    def _get_model(self):
        try:
            mclass = getattr(networks, 'VAE%s' % self.res_str)
        except AttributeError as excep:
            err_str = 'No network with resolution string %s defined.' % self.res_str
            raise AttributeError(err_str) from excep

        print("Using %s on %d GPUs" % (mclass.__name__,  torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(mclass(self.device, z_dim=self.z_dim, info=self.n_info))
        else:
            self.model = mclass(self.device, z_dim=self.z_dim, info=self.n_info)
        self.model.to(self.device)

    @staticmethod
    def _to_numpy(torch_arr):
        return torch_arr.detach().cpu().clone().numpy()

    def run(self, runtype):
        if runtype in ['train', 'all']:
            mu_train, logvar_train = self.train()

            label_all = self.preproc.label_r[:mu_train.shape[0]]
            with h5py.File(self.output_folder + '/data_train.h5', "w") as data_tem:
                data_tem['mu_all'] = mu_train
                data_tem['logvar_all'] = logvar_train
                data_tem['label_all'] = label_all

        if runtype in ['fit', 'all']:
            mu_fit, logvar_fit = self.fit()
            label_all = self.preproc.label_r[:mu_fit.shape[0]]
            with h5py.File(self.output_folder + '/data_fit.h5', "w") as data_tem:
                data_tem['mu_all'] = mu_fit
                data_tem['logvar_all'] = logvar_fit
                data_tem['label_all'] = label_all

        if runtype in ['generate', 'all']:
            generating(self.z_dim, self.n_info, self.res_str, self.output_folder, self.device)

        if runtype == 'all':
            if self.z_dim > 1:
                yvals = [mu_train, mu_fit]
            else:
                yvals = [label_all, label_all]

            plt.figure(figsize=(8, 4))
            plt.subplot(121)
            plt.scatter(mu_train[:,0], yvals[0][:,1], s=1, c=label_all[:,1])
            plt.title('after training')
            plt.subplot(122)
            plt.scatter(mu_fit[:,0], yvals[1][:,1], s=1, c=label_all[:,1])
            plt.title('after fitting')
            plt.savefig(self.output_folder + '/latent_space_gain.png')

    def _best_projection_slice(self, intens_3d, ind):
        select_plane = self.preproc.planes_s_th[ind]
        size = intens_3d.shape[2]
        grid = select_plane.float() #/(volume_size*sampling//2+1)
        return functional.grid_sample(intens_3d.view(1, 1, size, size, size),
                                      grid.view(1, size, size, 1, 3),
                                      mode='bilinear', padding_mode='zeros',
                                      align_corners=None)[0][0][:, :].reshape(size, size)

    def loss_function(self, recon_intens_3d, x, mu, logvar, b_num):
        recon_x = torch.zeros_like(x)
        for i in range(self.batch_size):
            arrth = recon_intens_3d[i]
            symarrth = 0.25 * (arrth + torch.flip(arrth, [1]) +
                               torch.flip(arrth, [2]) + torch.flip(arrth, [3]))
            symarrth = (symarrth + symarrth.transpose(1,3).transpose(2,3) +
                        symarrth.transpose(1,2).transpose(2,3)) / 3.
            symarrth = 0.5 * (symarrth + torch.flip(symarrth, [1, 2, 3]))
            recon_intens_3d_sym = symarrth
            recon_x_i = self._best_projection_slice(recon_intens_3d_sym, b_num*self.batch_size + i)
            recon_x[i] = recon_x_i

        BCE = functional.binary_cross_entropy(recon_x, x, reduction='sum')
        BSE = torch.sum((recon_x - x)**2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD + BSE, BCE, BSE, KLD, recon_x

    def train(self):
        print('Start training')
        intensize = self.preproc.intens.shape[2]
        print(self.n_epochs,' intens_size =', intensize)

        for epoch in range(self.n_epochs):
            mu_all_0 = np.zeros((1, self.z_dim))
            logvar_all_0 = np.zeros((1, self.z_dim))

            for i in range(self.preproc.intens.shape[0] // self.batch_size):
                # Local batches and labels
                intens_batch = self.preproc.intens[i*self.batch_size:(i+1)*self.batch_size]
                images = torch.from_numpy(intens_batch).view(self.batch_size, 1, intensize, intensize)
                images = images.float().to(self.device)

                ori_batch = self.preproc.rotation_sq_r[i*self.batch_size:(i+1)*self.batch_size]
                ori = torch.from_numpy(ori_batch).view(self.batch_size, 5)
                ori = ori.float().to(self.device)

                recon_images, mu, logvar = self.model([images, ori])
                loss, bce, bse, kld, recon_2D_x = self.loss_function(recon_images, images,
                                                                     mu, logvar, i)

                if (epoch = self.n_epochs - 1):
                    mu_all_0 = np.concatenate((mu_all_0, mu.detach().cpu().clone().numpy()),axis=0)
                    logvar_all_0 = np.concatenate((logvar_all_0, logvar.detach().cpu().clone().numpy()),axis=0)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i%100 == 0:
                    to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(epoch+1,
                                            self.n_epochs, loss.data.item(),
                                            bce.data.item(), bse.data.item(), kld.data.item())
                    print(to_print)

        torch.save(self.model.module.state_dict(), self.output_folder + '/Vae_CNN3D_dict')

        mu_all = mu_all_0[1:,:]
        logvar_all = logvar_all_0[1:,:]
        print('Done training!')
        return mu_all, logvar_all

    def fit(self):
        self.model.module.load_state_dict(torch.load(self.output_folder + '/Vae_CNN3D_dict'))
        self.model.eval()

        print('Start fitting')
        intensize = self.preproc.intens.shape[2]
        mu_all_0 = np.zeros((1, self.z_dim))
        logvar_all_0 = np.zeros((1, self.z_dim))
        recon_2D_all_0 = np.zeros((1,1,self.preproc.intens.shape[2],self.preproc.intens.shape[2]))
        print(self.z_dim, mu_all_0.shape)

        for i in range(self.preproc.intens.shape[0]//self.batch_size):
            # Local batches and labels
            intens_batch = self.preproc.intens[i*self.batch_size:(i+1)*self.batch_size]
            images = torch.from_numpy(intens_batch).view(self.batch_size, 1, intensize, intensize)
            images = images.float().to(self.device)

            ori_batch = self.preproc.rotation_sq_r[i*self.batch_size:(i+1)*self.batch_size]
            ori = torch.from_numpy(ori_batch).view(self.batch_size, 5)
            ori = ori.float().to(self.device)
            recon_images, mu, logvar = self.model([images, ori])
            loss, bce, bse, kld, recon_2D_x = self.loss_function(recon_images, images,
                                                                 mu, logvar, i)

            mu_all_0 = np.concatenate((mu_all_0, mu.detach().cpu().clone().numpy()),axis=0)
            logvar_all_0 = np.concatenate((logvar_all_0, logvar.detach().cpu().clone().numpy()),axis=0)


            if i < 3:
                recon_2D_all_0 = np.append(recon_2D_all_0, self._to_numpy(recon_2D_x))

            if i%self.batch_size == 0:
                print("Batch[{}/{}] Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(i*self.batch_size,
                                        10000, loss.data.item(),
                                        bce.data.item(), bse.data.item(),
                                        kld.data.item()))

        mu_all = mu_all_0[1:,:]
        logvar_all = logvar_all_0[1:,:]
        print('Done fitting!')

        return mu_all, logvar_all

def main():
    desc_str = 'Process 2D intensity averages using a Variation AutoEncoder framework'
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('config_file', help='Path to configuration file')
    parser.add_argument('-t', '--runtype', default='all',
                        help='Run type: all (default), fit, train, generate')
    parser.add_argument('-d', '--devnum', type=int, default=0,
                        help='Device index for multi-GPU nodes (default: 0)')
    args = parser.parse_args()

    if args.runtype not in ['all', 'fit', 'train', 'generate']:
        raise ValueError('Unknown run type: %s' % args.runtype)

    device = torch.device('cuda:%d'%args.devnum if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    recon = VAEReconstructor(args.config_file, device)

    torch.manual_seed(0)
    recon.run(args.runtype)

if __name__ == '__main__':
    main()
