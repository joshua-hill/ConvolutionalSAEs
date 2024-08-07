import torch
import h5py
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm


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
    
class HDF5ActivationsDataset(Dataset):
    def __init__(self, h5_file, data_key, transform=None):
        self.h5_file = h5py.File(h5_file, 'r')
        self.data = self.h5_file[data_key]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        
        if self.transform:
            x = self.transform(x)

        return torch.tensor(x, dtype=torch.float32)
    
def load_alexnet():
    model = torchvision.models.alexnet(weights='AlexNet_Weights.DEFAULT')
    model.eval()
    return model

def create_image_dataloader(batch_size=64, root='path/to/imagenet'):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
def create_activations_dataloader_from_h5(batch_size=64, root='alexnet_activations.h5', data_key='features.2'):
    dataset = HDF5ActivationsDataset(h5_file=root, data_key=data_key)
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

    for i, (inputs, _) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        with torch.no_grad():
            out = model.features(inputs)
        print(f'Batch {i + 1}/{len(dataloader)}')

        if (i + 1) % 50 == 0:  # Save activations incrementally every 50 batches
            save_activations(extractor.get_activations(), output_file)
            extractor.clear_activations()  # Clear activations to free up memory

    # Save any remaining activations
    save_activations(extractor.get_activations(), output_file)

def save_activations(activations, output_file):
    with h5py.File(output_file, 'a') as f:  # Open in append mode
        for layer_name, activation in activations.items():
            activation_np = activation.numpy()
            if layer_name in f:
                dataset = f[layer_name]
                dataset.resize((dataset.shape[0] + activation_np.shape[0]), axis=0)
                dataset[-activation_np.shape[0]:] = activation_np
            else:
                max_shape = (None,) + activation_np.shape[1:]  # None makes the first dimension resizable
                f.create_dataset(layer_name, data=activation_np, maxshape=max_shape, chunks=True)