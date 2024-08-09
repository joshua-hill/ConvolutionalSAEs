from csae import CSAE
import torch
from extract import ActivationExtractor, load_alexnet, create_image_dataloader
import torch
import torch.nn.functional as F

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

def test_extraction():
    # Test the ActivationExtractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    alexnet = load_alexnet().to(device)
    extractor = ActivationExtractor(alexnet, ['classifier.6'])

    test_input = torch.randn(8, 3, 224, 224).to(device)
    with torch.no_grad():
        output = alexnet(test_input)

    assert 'classifier.6' in extractor.get_activations().keys(), "Activation extraction failed"
    assert output.shape == extractor.get_activations()['classifier.6'].shape, "Activation shape mismatch"
    assert torch.allclose(output, extractor.get_activations()['classifier.6']), "Activation value mismatch"
    print("ActivationExtractor test passed successfully!")

if __name__ == '__main__':
    test_csae()
    test_extraction()