import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RegularGridInterpolator as rgi
import random
from pylab import cross,dot,inv
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy import interpolate
import os
import sys
import networks


def generating(z_dim, info, res, fold_save, device):
    def ave_fun(q, a, b, c):
             int = a*q**(-b) + c
             return int

    def rotation(x,y,z,axis,angle):
        a = angle
        mx = [[1, 0, 0],[0, np.cos(a), -np.sin(a)],[0, np.sin(a), np.cos(a)]]
        my = [[np.cos(a), 0, np.sin(a)],[0, 1, 0],[-np.sin(a), 0, np.cos(a)]]
        mz = [[np.cos(a), -np.sin(a), 0],[np.sin(a), np.cos(a), 0],[0, 0, 1]]
        Matrix_r = [mx,my,mz]
        M_axis = Matrix_r[axis]
        t = np.transpose(np.array([x,y,z]), (1,2,0))
        x1,y1,z1 = np.transpose(t @ M_axis, (2,0,1))
        return x1,y1,z1

    def generate_volumes(latent_class, fold_save, res, z_dim, info):
        if (res == 0):
            scale = 81
            half_scale = scale//2
            factor = 243/81.

        if (res == 1):
            scale = 161
            half_scale = scale//2
            factor = 243/161.

        if (res == 2):
            scale = 243
            half_scale = scale//2
            factor = 1.

        if (res == 10):
            scale = 81
            half_scale = scale//2
            factor = 243/161.


        x0,y0,z0 = np.indices((scale,scale,scale)); x0-=half_scale; y0-=half_scale; z0-=half_scale
        rad_float = np.sqrt(x0**2 + y0**2 +z0**2)*factor
        ave_3d = np.array([ave_fun(i,500,3.3,0.0005) for i in rad_float.ravel()]).reshape(scale,scale,scale)

        U = np.array([0, 0, 1])
        V011 = np.array([0, 1, 1])
        V111 = np.array([1, 1, 1])
        x001,y001 = np.indices((scale,scale)); x001-=half_scale; y001-=half_scale
        z001 = x001*0.0
        x011,y011,z011 = rotation(x001,y001,z001,0,np.pi/4)

        x111_0,y111_0,z111_0 = rotation(x001,y001,z001,0,0.955316618)
        x111,y111,z111 = rotation(x111_0,y111_0,z111_0,2,np.pi/4)

        #model = VAE(z_dim=z_dim, info=info).to(device)
        def _get_model(self):
            try:
                mclass = getattr(networks, 'VAE%s' % self.res_str)
            except AttributeError as excep:
                err_str = 'No network with resolution string %s defined.' % self.res_str
                raise AttributeError(err_str) from excep

            print("Using %s on %d GPUs" % (mclass.__name__,  torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                model = nn.DataParallel(mclass(device, z_dim=z_dim, info=info))
            else:
                model = mclass(device, z_dim=z_dim, info=info)
            model.to(device)

        torch.manual_seed(0)
        #optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.module.load_state_dict(torch.load(fold_save+'/Vae_CNN3D_dict'))
        model.eval()


        nv = latent_class.shape[0]
        rec_model_plane = np.zeros([nv,3,scale,scale])
        rec_model_3D_example = np.zeros((nv,scale,scale,scale))
        latent_class = latent_class
        latent_class_th = torch.from_numpy(latent_class).to(device).float()

        x0,y0,z0 = np.indices((scale,scale,scale))
        x0-=half_scale; y0-=half_scale; z0-=half_scale
        for i in range(nv):
            arrth = model.module.decode(latent_class_th[i])[0]
            symarrth = 0.25 * (arrth + torch.flip(arrth, [1]) + torch.flip(arrth, [2]) + torch.flip(arrth, [3]))
            symarrth = (symarrth + symarrth.transpose(1,3).transpose(2,3) + symarrth.transpose(1,2).transpose(2,3)) / 3.
            symarrth = 0.5 * (symarrth + torch.flip(symarrth, [1, 2, 3]))
            rec_model_3D = symarrth.detach().cpu().clone().numpy()
            rec_model_3D_example[i] = rec_model_3D[0]*ave_3d

            Interpolating_model_m = rgi((x0[:,0,0],y0[0,:,0],z0[0,0,:]), rec_model_3D[0]*ave_3d, bounds_error=False, fill_value=0.0)
            image_001 = Interpolating_model_m(np.array([x001,y001,z001]).T).ravel()
            image_011 = Interpolating_model_m(np.array([x011,y011,z011]).T).ravel()
            image_111 = Interpolating_model_m(np.array([x111,y111,z111]).T).ravel()
            rec_model_plane[i,0] = image_001.reshape(scale,scale)
            rec_model_plane[i,1] = image_011.reshape(scale,scale)
            rec_model_plane[i,2] = image_111.reshape(scale,scale)

            if (i % 10 == 0):
                print(i,'/200')

        return rec_model_3D_example, rec_model_plane


    squence_name = '/sequence_01/'
    try:
        os.mkdir(fold_save+squence_name)
    except:
        print('cannot creat dir')

    with h5py.File(fold_save+'/data_eval.h5', "r") as data_tem:
        mu_all = data_tem['mu_all'][:]
        logvar_all = data_tem['logvar_all'][:]
        label_all = data_tem['label_all'][:]

    if (z_dim == 1):
        zx_mean = np.mean(mu_all[:,0])
        stepx = (np.max(mu_all) - np.min(mu_all))/48
        zx = np.arange(48)
        zx = zx.ravel() * stepx + np.min(mu_all)
        zy = zx
        latent_class = np.array([zx]).T
        yy = np.zeros(48)+np.mean(label_all[:,1])
        fig = plt.figure(figsize=(8,5))
        plt.scatter(mu_all, label_all[:,1], c=label_all[:,1], s=5, alpha=0.35)
        plt.scatter(zx,yy, marker='+', color='black', s=30, alpha = 0.6)
        plt.xlabel('z')
        plt.ylabel('PCA-2')
        plt.colorbar(label='clpca-2')
        plt.tight_layout()
        plt.savefig(fold_save+squence_name+"/Latent_1D.png")

        rec_model_3D_example, rec_model_plane = generate_volumes(latent_class, fold_save, res, z_dim)

        Latent_data_density = np.zeros(48)
        for i in range(48):
            zxi = zx[i]
            n_cover = np.where((mu_all > zxi-stepx/2) & (mu_all < zxi+stepx/2))[0]
            Latent_data_density[i] = len(n_cover)

        ns = np.arange(48)

        fig = plt.figure(figsize=(20,5))
        for i in range(12):
            plt.subplot(3,12,1+i)
            plt.imshow(np.log10(rec_model_plane[i,0]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            if (i == 0):
                plt.ylabel('001')
            plt.subplot(3,12,1+i+12)
            plt.imshow(np.log10(rec_model_plane[i,1]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            if (i == 0):
                plt.ylabel('011')
            plt.subplot(3,12,1+i+24)
            plt.imshow(np.log10(rec_model_plane[i,2]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            plt.xlabel('z = '+str("%0.2f" % zx[i]))
            if (i == 0):
                plt.ylabel('111')
        fig.tight_layout()
        plt.savefig(fold_save+squence_name+"/sequence_intens_00.png")
        fig = plt.figure(figsize=(20,5))
        for i in range(12):
            plt.subplot(3,12,1+i)
            plt.imshow(np.log10(rec_model_plane[i+12,0]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            if (i == 0):
                plt.ylabel('001')
            plt.subplot(3,12,1+i+12)
            plt.imshow(np.log10(rec_model_plane[i+12,1]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            if (i == 0):
                plt.ylabel('011')
            plt.subplot(3,12,1+i+24)
            plt.imshow(np.log10(rec_model_plane[i+12,2]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            plt.xlabel('z = '+str("%0.2f" % zx[i+12]))
            if (i == 0):
                plt.ylabel('111')
        fig.tight_layout()
        plt.savefig(fold_save+squence_name+"/sequence_intens_01.png")
        fig = plt.figure(figsize=(20,5))
        for i in range(12):
            plt.subplot(3,12,1+i)
            plt.imshow(np.log10(rec_model_plane[i+24,0]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            if (i == 0):
                plt.ylabel('001')
            plt.subplot(3,12,1+i+12)
            plt.imshow(np.log10(rec_model_plane[i+24,1]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            if (i == 0):
                plt.ylabel('011')
            plt.subplot(3,12,1+i+24)
            plt.imshow(np.log10(rec_model_plane[i+24,2]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            plt.xlabel('z = '+str("%0.2f" % zx[i+24]))
            if (i == 0):
                plt.ylabel('111')
        fig.tight_layout()
        plt.savefig(fold_save+squence_name+"/sequence_intens_02.png")
        fig = plt.figure(figsize=(20,5))
        for i in range(12):
            plt.subplot(3,12,1+i)
            plt.imshow(np.log10(rec_model_plane[i+36,0]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            if (i == 0):
                plt.ylabel('001')
            plt.subplot(3,12,1+i+12)
            plt.imshow(np.log10(rec_model_plane[i+36,1]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            if (i == 0):
                plt.ylabel('011')
            plt.subplot(3,12,1+i+24)
            plt.imshow(np.log10(rec_model_plane[i+36,2]),vmax=0,vmin=-5.5 )
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            plt.xlabel('z = '+str("%0.2f" % zx[i+36]))
            if (i == 0):
                plt.ylabel('111')
        fig.tight_layout()
        plt.savefig(fold_save+squence_name+"/sequence_intens_03.png")

    if (z_dim == 2):
        zx_mean = (np.max(mu_all[:,0]) + np.min(mu_all[:,0]))//2
        zy_mean = (np.max(mu_all[:,1]) + np.min(mu_all[:,1]))//2 + 1
        stepx = (np.max(mu_all[:,0]) - np.min(mu_all[:,0]))/20
        stepy = (np.max(mu_all[:,1]) - np.min(mu_all[:,1]))/10
        zx_cent = np.int(-zx_mean/stepx + 10)
        zy_cent = np.int(-zy_mean/stepy + 5)
        zx, zy =  np.indices((20,10)); zx-=zx_cent; zy-=zy_cent
        zx = zx.ravel() * stepx
        zy = zy.ravel() * stepy
        latent_class = np.array([zx,zy]).T

        fig = plt.figure(figsize=(8,5))
        plt.scatter(mu_all[:,0], mu_all[:,1], c=label_all[:,1], s=5, alpha=0.35)
        plt.scatter(zx,zy, marker='+', color='black', s=30, alpha = 0.6)
        plt.xlabel('z-1')
        plt.ylabel('z-2')
        plt.colorbar(label='clpca-2')
        plt.tight_layout()
        plt.savefig(fold_save+squence_name+"/Latent_2D.png")

        rec_model_3D_example, rec_model_plane = generate_volumes(latent_class, fold_save, res, z_dim)

        Latent_data_density = np.zeros(200)
        for i in range(200):
            zxi = zx[i]
            zyi = zy[i]
            n_cover = np.where((mu_all[:,0] > zxi-0.225) & (mu_all[:,0] < zxi+0.225) & (mu_all[:,1] > zyi-0.15) & (mu_all[:,1] < zyi+0.15))[0]
            Latent_data_density[i] = len(n_cover)

        ns = np.arange(200)
        ns1 = ns.reshape(10,20).T
        ns1 = np.flip(ns1, axis=1)
        ns2 = ns1.ravel()
        plt.figure(figsize=(30,15))
        for i in range(200):
            ax = plt.subplot(10,20,ns2[i]+1)
            if (Latent_data_density[i] >= 1):
                plt.imshow(np.log10(rec_model_plane[i][0]),vmax=0,vmin=-5.5)
                ax.patch.set_alpha(1.0)
            if (Latent_data_density[i] < 1):
                plt.imshow(np.log10(rec_model_plane[i][0]),vmax=0,vmin=-5.5, alpha=0.65)
                ax.patch.set_alpha(0.5)
            plt.subplots_adjust(hspace = .001, wspace = .001)
            plt.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False)
            plt.tick_params(axis='y',which='both',left=False,right=False, labelleft=False)
            if ((ns2[i]+1)%20 == 1):
                plt.ylabel('z-2='+str("%0.2f" % zy[i]))
            if ((ns2[i]+1) > 180):
                plt.xlabel('z-1='+str("%0.2f" % zx[i]))
            #plt.xlabel('z=('+str("%0.2f" % zx[i]) +str("%0.2f" % zy[i]) +')')
        plt.tight_layout()
        plt.savefig(fold_save+squence_name+"/sequence_intens.png")




    h5 = h5py.File(fold_save + squence_name+'3D_rec_models.h5','w')
    h5['rec_model_3D_example'] = rec_model_3D_example
    h5['rec_model_planes'] = rec_model_plane
    h5['latent_class'] = latent_class
    h5['label_all'] = label_all
    h5['logvar_all'] = logvar_all
    h5['mu_all'] = mu_all
    h5['zx'] = zx
    h5['zy'] = zy
    h5.close()
