import torch
import torch.nn as nn


# Layer Utils
class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


def _conv3x3(in_channels, out_channels, stride=1, padding=0):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=padding
    )


def _conv4x4(in_channels, out_channels, stride=1, padding=0):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=4, stride=stride, padding=padding
    )


def _deconv3x3(in_channels, out_channels, stride=1, padding=0):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=padding
    )


def _deconv4x4(in_channels, out_channels, stride=1, padding=0):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=4, stride=stride, padding=padding
    )


def resize_conv3x3(in_channels, out_channels, scale=1, stride=1, padding=0):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return _conv3x3(in_channels, out_channels, stride, padding)
    return nn.Sequential(
        Interpolate(scale_factor=scale),
        _conv3x3(in_channels, out_channels, stride, padding),
    )


def resize_conv4x4(in_channels, out_channels, scale=1, stride=1, padding=0):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return _conv4x4(in_channels, out_channels, stride, padding)
    return nn.Sequential(
        Interpolate(scale_factor=scale),
        _conv4x4(in_channels, out_channels, stride, padding),
    )


class ConvEncoder(nn.Module):
    def __init__(
        self, inchannels, outchannels, latent_dim, stride=1, padding=0, downsample=None
    ):
        super().__init__()

        self.inchannels = inchannels
        self.outchannels = outchannels
        self.latent_dim = latent_dim
        self.stride = stride
        self.downsample = downsample
        self.padding = padding

        self.encoder = nn.Sequential(
            _conv4x4(inchannels, outchannels // 8, stride=2, padding=1),
            nn.BatchNorm2d(outchannels // 8),
            nn.ReLU(),
            _conv4x4(outchannels // 8, outchannels // 4, stride=2, padding=1),
            nn.BatchNorm2d(outchannels // 4),
            nn.ReLU(),
            _conv4x4(outchannels // 4, outchannels // 2, stride=2, padding=1),
            nn.BatchNorm2d(outchannels // 2),
            nn.ReLU(),
            _conv4x4(outchannels // 2, outchannels, stride=2, padding=1),
            nn.BatchNorm2d(outchannels),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        return x


class ConvDecoder(nn.Module):
    def __init__(
        self, inchannels, outchannels, latent_dim, stride=1, padding=0, downsample=None
    ):
        super().__init__()

        self.inchannels = inchannels
        self.outchannels = outchannels
        self.latent_dim = latent_dim
        self.stride = stride
        self.downsample = downsample
        self.padding = padding

        self.decoder = nn.Sequential(
            _deconv4x4(outchannels, outchannels // 2, stride=2, padding=1),
            nn.BatchNorm2d(outchannels // 2),
            nn.ReLU(),
            _deconv4x4(outchannels // 2, outchannels // 4, stride=2, padding=1),
            nn.BatchNorm2d(outchannels // 4),
            nn.ReLU(),
            _deconv4x4(outchannels // 4, outchannels // 8, stride=2, padding=1),
            nn.BatchNorm2d(outchannels // 8),
            nn.ReLU(),
            _deconv4x4(outchannels // 8, inchannels, stride=2, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.Sigmoid(),
        )

        self.linear = nn.Linear(latent_dim, outchannels * 2 * 2)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.outchannels, 2, 2)

        out = self.decoder(x)
        return out


def conv_encoder(inchannels, outchannels, latent_dim, stride=1, padding=0):
    return ConvEncoder(inchannels, outchannels, latent_dim, stride, padding)


def conv_decoder(inchannels, outchannels, latent_dim, stride=1, padding=0):
    return ConvDecoder(inchannels, outchannels, latent_dim, stride, padding)


class ConvVAE(nn.Module):
    def __init__(
        self, inchannels, outchannels, latent_dim, stride=1, padding=0, downsample=None
    ):
        super().__init__()

        self.inchannels = inchannels
        self.outchannels = outchannels
        self.latent_dim = latent_dim
        self.stride = stride
        self.downsample = downsample
        self.padding = padding

        # Encoder
        self.encode = conv_encoder(inchannels, outchannels, latent_dim, stride, padding)
        # Decoder
        self.decode = conv_decoder(inchannels, outchannels, latent_dim, stride, padding)

        # Distribution parameters
        self.fc_mu = nn.Linear(outchannels * 2 * 2, latent_dim)
        self.fc_var = nn.Linear(outchannels * 2 * 2, latent_dim)

    def reparameterize(self, mu, logvar):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space

        """
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return z

    def forward(self, x):
        x = self.encode(x)

        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, logvar)

        out = self.decode(z)

        return out, z, mu, logvar
