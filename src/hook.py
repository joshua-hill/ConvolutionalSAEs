import torch 
import torch.nn as nn



def get_activation(name, act_dict):
    def hook(model, input, output):
        act_dict[name].append(output.detach())
    return hook

def register_hooks(model, act_dict):
    for name, layer in model.named_modules():
        if name == 'features.2' or name == 'features.5' or name == 'features.7' or name == 'features.9' or name == 'features.12' or name=='avgpool':
            layer.register_forward_hook(get_activation(name, act_dict))

def register_replace_csae_hook(model, name, csae):
    model._modules[name].register_forward_hook(replace_activation_with_csae(model, csae))

def replace_activation_with_csae(model, csae):
    def hook(model, input, output):
        return csae.forward(input)[0]
    return hook

