import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class CSAE(nn.Module):
    def __init__(self, in_channels, dict_size, kernel_size=1, stride=1, padding=0):
        super(CSAE, self).__init__()
        
        self.encoder = nn.Conv2d(in_channels, dict_size, kernel_size, stride, padding, bias=True)
        self.decoder = weight_norm(nn.Conv2d(dict_size, in_channels, kernel_size, stride, padding, bias=True))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight.data)

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

def test_csae():
    # Test the CSAE
    batch_size, in_channels, height, width = 32, 64, 56, 56
    x = torch.randn(batch_size, in_channels, height, width)
    
    csae = CSAE(in_channels, dict_size=128)
    x_recon, z = csae(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Encoded shape: {z.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    
    # Check if shapes match
    assert x.shape == x_recon.shape, "Input and reconstruction shapes don't match"
    assert z.shape[1] == 128, "Encoded shape doesn't match the specified dictionary size"
    
    print("CSAE test passed successfully!")


