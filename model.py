import torch
import torch.nn as nn
# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 3, 3, 3)

class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=128, z_dim=1):
        super(VAE, self).__init__()
        self.encoder = None
        self.decoder = None
        self.z_dim = z_dim
        #self.x = x
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
        self.fc1 = nn.Linear(68, self.z_dim)
        self.fc2 = nn.Linear(68, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 64)
        #self.fc4 = nn.Linear(64, 128)



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
            #nn.ConvTranspose3d(8, 4, kernel_size=3, padding=1 ),
            #nn.BatchNorm3d(4), nn.ReLU(), nn.Dropout(p=0.3),
            nn.ConvTranspose3d(8, 1, kernel_size=3, padding=1 ),
            nn.Sigmoid()
            )

    def sampling(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(device)
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def latent_layer(self, h):
        h1 = torch.cat([h, self.ori[:,:4]], dim=1)
        mu, logvar = self.fc1(h1), self.fc2(h1)
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

    def forward(self, x_all):
        x = x_all[0]
        self.ori = x_all[1]
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
