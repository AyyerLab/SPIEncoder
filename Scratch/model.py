import torch
import torch.nn as nn
# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


class UnFlatten(nn.Module):
    def forward(self, x):
        return x.view(-1, 256, 3, 3, 3)

class VAE(nn.Module):
    def __init__(self, z_dim=1):
        super(VAE, self).__init__()
        self.encoder = None
        self.decoder = None
        self.z_dim = z_dim
        #self.x = x
        self._init_encoder()
        self._init_latent_layers()
        self._init_decoder()
        self.ori = None

    def _init_encoder(self, dropouts=False):
        modules = []
        layer_widths = [1, 16, 64, 128]

        # Convolutional layers
        for i in range(len(layer_widths) - 1):
            modules.extend([
                nn.Conv2d(layer_widths[i], layer_widths[i+1], kernel_size=3, stride=3, padding=1),
                nn.BatchNorm2d(layer_widths[i+1]),
                nn.ReLU()
            ])
            if dropouts:
                modules.append(nn.Dropout(p=0.1))

        # Flat layers
        modules.extend([
            nn.Flatten(),
            nn.Linear(3*3*128, 64),
            nn.ReLU()
        ])

        self.encoder = nn.Sequential(*tuple(modules))

    def _init_latent_layers(self):
        #self.fc0 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(68, self.z_dim)
        self.fc2 = nn.Linear(68, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 64)
        #self.fc4 = nn.Linear(64, 128)

    def _init_decoder(self, dropouts=False):
        layer_widths = [256, 128, 64, 16, 8]

        # Initial flat layers
        modules = [nn.Linear(64, 256*3*3*3), nn.ReLU(), UnFlatten()]

        # Convolutional layers
        for i in range(len(layer_widths) - 1):
            modules.extend([
                nn.ConvTranspose3d(layer_widths[i], layer_widths[i+1], kernel_size=3, stride=3),
                nn.BatchNorm3d(layer_widths[i+1]),
                nn.ReLU()
            ])
            if dropouts:
                modules.append(nn.Dropout(p=0.1))

        # Last sigmoid layer
        modules.extend([
            nn.ConvTranspose3d(layer_widths[-1], 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ])

        self.decoder = nn.Sequential(*tuple(modules))

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
