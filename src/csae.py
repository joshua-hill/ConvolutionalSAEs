import torch
import torch.nn as nn
import torch.nn.functional as F


class CSAE(nn.Module):
    def __init__(self, in_channels, dict_size, kernel_size=1, stride=1, padding=0):
        super(CSAE, self).__init__()
        
        self.encoder = nn.Conv2d(in_channels, dict_size, kernel_size, stride, padding, bias=True)
        nn.init.constant_(self.encoder.bias, 0)
        self.decoder = nn.Conv2d(dict_size, in_channels, kernel_size, stride, padding, bias=True)
        nn.init.constant_(self.decoder.bias, 0)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.decoder.weight)

        # Make the encoder weights the transpose of the decoder weights - as per Anthropic's suggestion
        self.encoder.weight.data = self.decoder.weight.data.transpose(0, 1)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class CSAEFactory:
    @staticmethod
    def create_csae_for_layer(layer_shape, dict_factor=4):
        in_channels = layer_shape[1]  # Assuming shape is [B, C, H, W]
        dict_size = in_channels * dict_factor
        return CSAE(in_channels, dict_size)




