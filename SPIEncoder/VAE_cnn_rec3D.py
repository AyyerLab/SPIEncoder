import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py
%matplotlib
import random
#source activate /home/ayyerkar/.conda/envs/cuda

#===========Input_image_pipline============
h5 = h5py.File('input_image_hd0.h5', 'r')
intens_input0 = h5['intens_input'][:]
labels = h5['labels'][:]
h5.close()

# Generate 'n' unique random numbers within a range
randomList = random.sample(range(0, 15000), 15000)

intens_input = intens_input0[:,::3,::3]
intens_input_r = intens_input[randomList]
label_r = labels[randomList]

intens = np.zeros_like(intens_input_r)
for i in range(intens_input_r.shape[0]):
    intens[i] = (intens_input_r[i]/np.max(intens_input_r[i]))*1.0

batch_size = 50


# Device configuration
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')



#===========Generate 2D tomograph planes============
from scipy.interpolate import RegularGridInterpolator as rgi
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

#x0,y0,z0 = np.indices((253,253,253)); x0-=126; y0-=126; z0-=126
x,y = np.indices((81,81)); x-=40; y-=40
R3d = 100000/3
z =   R3d - np.sqrt(R3d**2 - (x**2+y**2)) #ewald sphere
q = np.sqrt(x*x + y*y)
pi = np.pi


a1,a2,a3 = np.indices((10,10,10))
angel1 = a1.reshape(-1); angel2 = a2.reshape(-1); angel3 = a3.reshape(-1)
angle_all = np.array([angel3,angel2,angel1]).T/10.0*np.pi

n_planes = angle_all.shape[0]
planes_all = np.zeros((1000,81,81,3))

for i in range(n_planes):
    pi = np.pi
    r0, r1, r2 = angle_all[i]
    x1,y1,z1 = rotation(x,y,z,0,r0)
    x2,y2,z2 = rotation(x1,y1,z1,1,r1)
    x3,y3,z3 = rotation(x2,y2,z2,2,r2)
    planes_all[i] = np.array([x3,y3,z3]).T



#planes_all_r = planes_all.reshape(1000,81,81,3)
planes_all_th = torch.from_numpy(planes_all).to(device)




#====================VAE_CNN_3D recon=======================



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1,1152)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 128, 3, 3, 3)

class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=128, z_dim=3):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Conv2d(16, 64, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(3*3*128, 64),
            nn.ReLU()
        )
        #self.fc0 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, z_dim)
        self.fc2 = nn.Linear(64, z_dim)
        self.fc3 = nn.Linear(z_dim, 64)
        #self.fc4 = nn.Linear(64, 128)

        self.decoder = nn.Sequential(
            #nn.Linear(3, 64),
            nn.Linear(64, 128*3*3*3),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=3),
            nn.BatchNorm3d(64), nn.ReLU(), nn.Dropout(p=0.5),
            nn.ConvTranspose3d(64, 16, kernel_size=3, stride=3),
            nn.BatchNorm3d(16), nn.ReLU(), nn.Dropout(p=0.5),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=3),
            nn.Sigmoid()
        )

        self.planes_all_th = planes_all_th



    def sampling(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(device)
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def latent_layer(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.sampling(mu, logvar).to(device)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.latent_layer(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)

        return z, mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_intens_3d, x, mu, logvar):
    def best_projection_layer(recon_intens_3d_i, x_i, num=1000):
        select_planes = planes_all_th#[torch.randint(low=0,high=999,size=(num,))]
        grid = select_planes.float()/(253/3)
        tomo = torch.ones([num, 81, 81], dtype=torch.float)
        tm_indices = torch.ones([num], dtype=torch.float)
        for i in range(num):
            tm = F.grid_sample(recon_intens_3d_i.view(1,1,81,81,81), grid[i].view(1,81,81,1,3), mode='bilinear', padding_mode='zeros', align_corners=None)[0][0][:,:].reshape(81,81)
            tomo[i] = tm
            tm_indices[i] = torch.sum((tm - x_i)*(tm - x_i))

        recon_x_i = tomo[torch.argmin(tm_indices)]
        return recon_x_i

    recon_x = torch.zeros_like(x)
    for i in range(n_batches):
        recon_x[i] = best_projection_layer(recon_intens_3d[i], x[i])

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

epochs = 30
n_batches = 100

# Train model
for epoch in range(epochs):
    mu_all = np.zeros((1,3))
    logvar_all = np.zeros((1,3))
    for i in range(20):
        # Local batches and labels
        images = torch.from_numpy(intens[i*n_batches:(i+1)*n_batches]).view(100,1,81,81)
        images = images.float().to(device)
        recon_images, mu, logvar = model(images)
        mu_all = np.concatenate((mu_all, mu.detach().cpu().clone().numpy()),axis=0)
        logvar_all = np.concatenate((logvar_all, logvar.detach().cpu().clone().numpy()),axis=0)
        loss, bce, kld = loss_function(recon_images, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1,
                                epochs, loss.data.item(), bce.data.item(), kld.data.item())
        print(to_print)


#torch.save(model, 'pytorch_vae_cnn3D_model00_LD')

torch.save(model.state_dict(), 'pytorch_vae_cnn3D_model00_LD_dict')

R3d = 100000/3
z =   R3d - np.sqrt(R3d**2 - (x**2+y**2)) #ewald sphere
q = np.sqrt(x*x + y*y)
pi = np.pi


a1,a2,a3 = np.indices((20,20,20))
angel1 = a1.reshape(-1); angel2 = a2.reshape(-1); angel3 = a3.reshape(-1)
angle_all = np.array([angel3,angel2,angel1]).T/20.0*np.pi

n_planes = angle_all.shape[0]
planes_all = np.zeros((8000,81,81,3))

for i in range(n_planes):
    pi = np.pi
    r0, r1, r2 = angle_all[i]
    x1,y1,z1 = rotation(x,y,z,0,r0)
    x2,y2,z2 = rotation(x1,y1,z1,1,r1)
    x3,y3,z3 = rotation(x2,y2,z2,2,r2)
    planes_all[i] = np.array([x3,y3,z3]).T



planes_all_r = planes_all.reshape(8000,81,81,3)
planes_all_th = torch.from_numpy(planes_all)




#====================VAE_CNN_3D recon=======================


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1,1152)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 128, 3, 3, 3)

class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=128, z_dim=3):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Conv2d(16, 64, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(3*3*128, 64),
            nn.ReLU()
        )
        #self.fc0 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, z_dim)
        self.fc2 = nn.Linear(64, z_dim)
        self.fc3 = nn.Linear(z_dim, 64)
        #self.fc4 = nn.Linear(64, 128)

        self.decoder = nn.Sequential(
            #nn.Linear(3, 64),
            nn.Linear(64, 128*3*3*3),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=3),
            nn.BatchNorm3d(64), nn.ReLU(), nn.Dropout(p=0.5),
            nn.ConvTranspose3d(64, 16, kernel_size=3, stride=3),
            nn.BatchNorm3d(16), nn.ReLU(), nn.Dropout(p=0.5),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=3),
            nn.Sigmoid()
        )

        self.planes_all_th = planes_all_th



    def sampling(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def latent_layer(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.sampling(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.latent_layer(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)

        return z, mu, logvar


model = VAE()#.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_intens_3d, x, mu, logvar):
    def best_projection_layer(recon_intens_3d_i, x_i, num=100):
        select_planes = planes_all_th[torch.randint(low=0,high=8000,size=(num,))]
        grid = select_planes.float()/(253/3)
        tomo = torch.ones([num, 81, 81], dtype=torch.float)
        tm_indices = torch.ones([num], dtype=torch.float)
        for i in range(num):
            tm = F.grid_sample(recon_intens_3d_i.view(1,1,81,81,81), grid[i].view(1,81,81,1,3), mode='bilinear', padding_mode='zeros', align_corners=None)[0][0][:,:].reshape(81,81)
            tomo[i] = tm
            tm_indices[i] = torch.sum((tm - x_i)*(tm - x_i))

        recon_x_i = tomo[torch.argmin(tm_indices)]
        return recon_x_i

    recon_x = torch.zeros_like(x)
    for i in range(n_batches):
        recon_x[i] = best_projection_layer(recon_intens_3d[i], x[i])

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

epochs = 30
n_batches = 100

# Train model
for epoch in range(epochs):
    mu_all = np.zeros((1,3))
    logvar_all = np.zeros((1,3))
    for i in range(20):
        # Local batches and labels
        images = torch.from_numpy(intens[i*n_batches:(i+1)*n_batches]).view(100,1,81,81)
        images = images.float()
        recon_images, mu, logvar = model(images)
        mu_all = np.concatenate((mu_all, mu.detach().numpy()),axis=0)
        logvar_all = np.concatenate((logvar_all, logvar.detach().numpy()),axis=0)
        loss, bce, kld = loss_function(recon_images, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1,
                                epochs, loss.data.item(), bce.data.item(), kld.data.item())
        print(to_print)

