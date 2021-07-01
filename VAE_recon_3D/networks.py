import torch
import torch.nn as nn
# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 3, 3, 3)

class VAE_HD(nn.Module):
    def __init__(self, device, image_channels=1, h_dim=128, z_dim=2, info=1):
        super(VAE_HD, self).__init__()
        self.encoder = None
        self.decoder = None
        self.device = device
        self.z_dim = z_dim
        if (info == 0):
            self.info_dim = 4
        if (info > 0):
            self.info_dim = 5

        self._init_encoder()
        self._init_latent_layers()
        self._init_decoder()
        self.ori = None

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

    def _init_latent_layers(self):
        #self.fc0 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64+self.info_dim, self.z_dim)
        self.fc2 = nn.Linear(64+self.info_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 64)

# Dout=(Din −1)×stride[0]−2×padding[0]+(kernel_size[0]−1)+output_padding[0]+1
    def _init_decoder(self):
        self.decoder = nn.Sequential(
            #nn.Linear(3, 64),
            nn.Linear(64, 256*3*3*3),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=3),       #Din = 3*3*3   Dout = 9*9*9
            nn.BatchNorm3d(128), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1),      #Din = 9*9*9   Dout = 9*9*9
            nn.BatchNorm3d(64), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=3),      #Din = 9*9*9   Dout = 27*27*27
            nn.BatchNorm3d(32), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=3),      #Din = 27*27*27   Dout = 81*81*81
            nn.BatchNorm3d(16), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=3, padding=0),      #Din = 81*81*81   Dout = 243*243*243
            nn.BatchNorm3d(8), nn.ReLU(), #nn.Dropout(p=0.1),
            nn.ConvTranspose3d(8, 1, kernel_size=3, padding=1 ),      #Din = 243*243*243   Dout = 243*243*243
            nn.Sigmoid()
            )

    def sampling(self, mu, logvar):
        #std = logvar.mul(0.5).exp_().to(self.device)
        #esp = torch.randn(*mu.size()).to(self.device)
        #z = mu + std * esp
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        #return z

    def latent_layer(self, h):
        h1 = torch.cat([h, self.ori[:,:self.info_dim]], dim=1)
        mu, logvar = self.fc1(h1), self.fc2(h1)
        z = self.sampling(mu, logvar)#.to(self.device)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.latent_layer(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x_all):
        x = x_all[0]
        self.ori = x_all[1]
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

# middle resolution
class VAE_MD(nn.Module):
    def __init__(self, device, image_channels=1, h_dim=128, z_dim=2, info=1):
        super(VAE_MD, self).__init__()
        self.encoder = None
        self.decoder = None
        self.device = device
        self.z_dim = z_dim
        if (info == 0):
            self.info_dim = 4
        if (info > 0):
            self.info_dim = 5

        self._init_encoder()
        self._init_latent_layers()
        self._init_decoder()
        self.ori = None

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

    def _init_latent_layers(self):
        #self.fc0 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64+self.info_dim, self.z_dim)
        self.fc2 = nn.Linear(64+self.info_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 64)

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

    def sampling(self, mu, logvar):
        #std = logvar.mul(0.5).exp_().to(self.device)
        #esp = torch.randn(*mu.size()).to(self.device)
        #z = mu + std * esp
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        #return z

    def latent_layer(self, h):
        h1 = torch.cat([h, self.ori[:,:self.info_dim]], dim=1)
        mu, logvar = self.fc1(h1), self.fc2(h1)
        z = self.sampling(mu, logvar)#.to(self.device)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.latent_layer(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x_all):
        x = x_all[0]
        self.ori = x_all[1]
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


class VAE_LD(nn.Module):
    def __init__(self, image_channels=1, h_dim=128, z_dim=2, info=1):
        super(VAE_LD, self).__init__()
        self.encoder = None
        self.decoder = None
        self.z_dim = z_dim
        if (info == 0):
            self.info_dim = 4
        if (info > 0):
            self.info_dim = 5

        self._init_encoder()
        self._init_latent_layers()
        self._init_decoder()
        self.ori = None

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

    def _init_latent_layers(self):
        #self.fc0 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64+self.info_dim, self.z_dim)
        self.fc2 = nn.Linear(64+self.info_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 64)

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

    def sampling(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def latent_layer(self, h):
        h1 = torch.cat([h, self.ori[:,:self.info_dim]], dim=1)
        mu, logvar = self.fc1(h1), self.fc2(h1)
        z = self.sampling(mu, logvar)#.to(device)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.latent_layer(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x_all):
        x = x_all[0]
        self.ori = x_all[1]
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
