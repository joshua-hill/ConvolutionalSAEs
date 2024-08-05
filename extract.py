import torch
import h5py
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset


class ActivationExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.activations = {layer: [] for layer in target_layers}
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(layer_name):
            def forward_hook(module, input, output):
                self.activations[layer_name].append(output.detach().cpu())
            return forward_hook

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(hook_fn(name))

    def clear_activations(self):
        for layer in self.target_layers:
            self.activations[layer].clear()

    def get_activations(self):
        return {layer: torch.cat(acts, dim=0) for layer, acts in self.activations.items()}
    

def load_alexnet():
    model = torchvision.models.alexnet(pretrained=True)
    model.eval()
    return model

def create_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageNet(root='path/to/imagenet', split='train', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    

def extract_activations(model, dataloader, num_batches=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    target_layers = [
        'features.0',  # Conv1
        'features.3',  # Conv2
        'features.6',  # Conv3
        'features.8',  # Conv4
        'features.10'  # Conv5
    ]

    extractor = ActivationExtractor(model, target_layers)

    for i, (inputs, _) in enumerate(dataloader):
        if i >= 2:
            break
        inputs = inputs.to(device)
        with torch.no_grad():
            # Forward pass only through convolutional layers
            _ = model.features(inputs)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} batches")

    return extractor.get_activations()

def save_activations(activations, output_file):
    with h5py.File(output_file, 'w') as f:
        for layer_name, activation in activations.items():
            f.create_dataset(layer_name, data=activation.numpy())