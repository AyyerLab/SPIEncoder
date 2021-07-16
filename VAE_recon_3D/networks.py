import numpy as np
from scipy import interpolate
import torch
import torch.nn as nn

class UnFlatten(nn.Module):
    def forward(self, input_arr):
        return input_arr.view(-1, 256, 3, 3, 3)

# Dout=(Din −1)×stride[0]−2×padding[0]+(kernel_size[0]−1)+output_padding[0]+1

class BaseVAE(nn.Module):
    def __init__(self, device, z_dim=2, info=1):
        super().__init__()
        self.device = device
        self.z_dim = z_dim
        if info == 0:
            self.info_dim = 4
        if info > 0:
            self.info_dim = 5

        self._init_encoder()
        self._init_latent_layers()
        self._init_decoder()
        self.ori = None

    def _init_encoder(self):
        raise NotImplementedError('Subclass needs to implement _init_encoder')

    def _init_decoder(self):
        raise NotImplementedError('Subclass needs to implement _init_decoder')

    def preproc_sample_intens(self, intens_input):
        raise NotImplementedError('Subclass needs to implement preproc_sample_intens')

    def preproc_scale_down_plane(self, input_plane):
        raise NotImplementedError('Subclass needs to implement preproc_scale_down_plane')

    def _init_latent_layers(self):
        self.fc1 = nn.Linear(64+self.info_dim, self.z_dim)
        self.fc2 = nn.Linear(64+self.info_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 64)

    def encode(self, x):
        z, mu, logvar = self.latent_layer(self.encoder(x))
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def latent_layer(self, h):
        h1 = torch.cat([h, self.ori[:,:self.info_dim]], dim=1)
        mu, logvar = self.fc1(h1), self.fc2(h1)
        z = self.sampling(mu, logvar)#.to(device)
        return z, mu, logvar

    @staticmethod
    def sampling(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x_all):
        x = x_all[0]
        self.ori = x_all[1]
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    @staticmethod
    def _mask_circle(intens_input_c):
        print('input image size = ', intens_input_c.shape)
        imagesize = intens_input_c.shape[-1]
        ind = np.arange(imagesize) - imagesize//2
        x, y = np.meshgrid(ind, ind, indexing='ij')
        intrad_2d = np.sqrt(x**2 + y**2)
        intens_input_c[:,intrad_2d > imagesize//2] = 0.0

class VAEHD(BaseVAE):
    def _init_encoder(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(),# nn.Dropout(p=0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64), nn.ReLU(),# nn.Dropout(p=0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(128), nn.ReLU(),# nn.Dropout(p=0.1),
            nn.Flatten(),
            nn.Linear(3*3*128, 64),
            nn.ReLU()
            )

    def _init_decoder(self):
        self.decoder = nn.Sequential(
            nn.Linear(64, 256*3*3*3),
            nn.ReLU(),
            UnFlatten(),
            #Din = 3*3*3   Dout = 9*9*9
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=3),
            nn.BatchNorm3d(128), nn.ReLU(), #nn.Dropout(p=0.1),
            #Din = 9*9*9   Dout = 9*9*9
            nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(), #nn.Dropout(p=0.1),
            #Din = 9*9*9   Dout = 27*27*27
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=3),
            nn.BatchNorm3d(32), nn.ReLU(), #nn.Dropout(p=0.1),
            #Din = 27*27*27   Dout = 81*81*81
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=3),
            nn.BatchNorm3d(16), nn.ReLU(), #nn.Dropout(p=0.1),
            #Din = 81*81*81   Dout = 243*243*243
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm3d(8), nn.ReLU(), #nn.Dropout(p=0.1),
            #Din = 243*243*243   Dout = 243*243*243
            nn.ConvTranspose3d(8, 1, kernel_size=3, padding=1 ),
            nn.Sigmoid()
            )

    def preproc_sample_intens(self, intens_input):
        intens_input_c = intens_input
        self._mask_circle(intens_input_c)
        return intens_input_c

    def preproc_scale_down_plane(self, input_plane):
        return input_plane / (input_plane.shape[1]//2+1)

class VAEMD(BaseVAE):
    def _init_encoder(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(),# nn.Dropout(p=0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64), nn.ReLU(),# nn.Dropout(p=0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(128), nn.ReLU(),# nn.Dropout(p=0.1),
            nn.Flatten(),
            nn.Linear(3*3*128, 64),
            nn.ReLU()
            )

    def _init_decoder(self):
        self.decoder = nn.Sequential(
            #nn.Linear(3, 64),
            nn.Linear(64, 256*3*3*3),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=3),
            nn.BatchNorm3d(128), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1 ),
            nn.BatchNorm3d(64), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=3),
            nn.BatchNorm3d(32), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=3),
            nn.BatchNorm3d(16), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(8, 1, kernel_size=3, padding=1 ),
            nn.Sigmoid()
            )

    def preproc_sample_intens(self, intens_input):
        num_l = intens_input.shape[0]
        x_orig = np.arange(-121, 122, 1.) / 121.
        y_orig = np.arange(-121, 122, 1.) / 121.

        lmax = 80

        x_new = np.arange(-lmax, lmax+1, 1.) / 121
        y_new = np.arange(-lmax, lmax+1, 1.) / 121
        intens_input_c = np.zeros([num_l, 2*lmax+1, 2*lmax+1])
        for i in range(num_l):
            input_map = intens_input[i]
            interpf = interpolate.interp2d(x_orig, y_orig, input_map, kind='cubic')
            intens_input_c[i] = interpf(x_new, y_new)

        self._mask_circle(intens_input_c)

        return intens_input_c

    def preproc_scale_down_plane(self, input_plane):
        x_orig = np.arange(-121, 122, 1.) / 121
        y_orig = np.arange(-121, 122, 1.) / 121
        interpf = interpolate.interp2d(x_orig, y_orig, input_plane, kind='cubic')

        x_new = np.arange(-80, 81, 1.) / 121.
        y_new = np.arange(-80, 81, 1.) / 121.
        return interpf(x_new, y_new)/(len(x_new)//2+1)  #MD

class VAELD(BaseVAE):
    def _init_encoder(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.Conv2d(16, 64, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),# nn.Dropout(p=0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),# nn.Dropout(p=0.1),
            nn.Flatten(),
            nn.Linear(3*3*128, 64),
            nn.ReLU()
            )

    def _init_decoder(self):
        self.decoder = nn.Sequential(
            #nn.Linear(3, 64),
            nn.Linear(64, 256*3*3*3),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=3),
            nn.BatchNorm3d(128), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1 ),
            nn.BatchNorm3d(64), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(64, 16, kernel_size=3, stride=3),
            nn.BatchNorm3d(16), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=3),
            nn.BatchNorm3d(8), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(8, 1, kernel_size=3, padding=1 ),
            nn.Sigmoid()
            )

    def preproc_sample_intens(self, intens_input):
        num_l = intens_input.shape[0]
        x_orig = np.arange(-121, 122, 1.) / 121.
        y_orig = np.arange(-121, 122, 1.) / 121.

        lmax = 40

        x_new = np.arange(-lmax, lmax+1, 1.) / 121
        y_new = np.arange(-lmax, lmax+1, 1.) / 121
        intens_input_c = np.zeros([num_l, 2*lmax+1, 2*lmax+1])
        for i in range(num_l):
            input_map = intens_input[i]
            interpf = interpolate.interp2d(x_orig, y_orig, input_map, kind='cubic')
            intens_input_c[i] = interpf(x_new,y_new)

        self._mask_circle(intens_input_c)

        return intens_input_c

    def preproc_scale_down_plane(self, input_plane):
        x_orig = np.arange(-121, 122, 1.) / 121
        y_orig = np.arange(-121, 122, 1.) / 121
        interpf = interpolate.interp2d(x_orig, y_orig, input_plane, kind='cubic')

        x_new = np.arange(-40, 41, 1.) / 121.
        y_new = np.arange(-40, 41, 1.) / 121.
        return interpf(x_new, y_new)/(len(x_new)//2+1)  #MD

class VAELDPaper(VAELD):
    def preproc_sample_intens(self, intens_input):
        intens_input_c = intens_input[:,42:204:2,42:204:2]
        self._mask_circle(intens_input_c)
        return intens_input_c

    def preproc_scale_down_plane(self, input_plane):
        x_orig = np.arange(-121, 122, 1.) / 121
        y_orig = np.arange(-121, 122, 1.) / 121
        interpf = interpolate.interp2d(x_orig, y_orig, input_plane, kind='cubic')

        x_new = np.arange(-40, 41, 1) / 60.
        y_new = np.arange(-40, 41, 1) / 60.
        return interpf(x_new, y_new) / (len(x_new)*2//2+1)
