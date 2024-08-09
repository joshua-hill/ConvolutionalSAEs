import torch
from torch import nn as nn
from torch.nn import functional as F
from csae import CSAE
from tqdm import tqdm
import os
from extract import load_alexnet, ActivationExtractor, create_image_dataloader
from hook import register_replace_csae_hook, replace_activation_with_csae

def evaluate_kl_div(csae, args, dataloader):
    # want to run imagenet through alexnet, get the logits. Then run imagenet through alexnet where activations are replaced with csae activations, measure KL divergence as the loss for reconstructio naccuracy. Also want to Measure the average L0 and L1 sparsity of the activations
    #technically, the feature is the intermediate_activation * the norm of the corresponding column in decoder, 

    kl_div_list =[]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csae = csae.to(device)
    csae.eval()

    dataloader = create_image_dataloader(batch_size=512)

    # Load the AlexNet model
    an = load_alexnet().to(device)
    an_2 = load_alexnet().to(device)

    #register hooks on second model instance
    register_replace_csae_hook(an_2, args['target_layer'], csae)

    pbar = tqdm(enumerate(dataloader))

    for i, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Get the output distribution from the AlexNet model
        logits = an.forward(inputs)

        #Get the output distribution form AlexNet model with activations replaced with csae
        logits_2 = an_2(inputs)

        # Calculate the KL divergence between the logits
        kl_div = F.kl_div(F.log_softmax(logits_2, dim=1), F.softmax(logits, dim=1))
        #should make it so that we can concat all the kl_divs along the 0th batch dimension, and then find mean, instead of finding the mean of each element of the lsit,t hen the list itself.
        kl_div_list.append(kl_div.item())

    return kl_div_list
    
   
def evaluate_sparsity(csae, args, dataloader):

    l0_sparsity_list = []
    l1_sparsity_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csae = csae.to(device)
    csae.eval()

    dataloader = create_image_dataloader(batch_size=512)
    total_iterations = len(dataloader)

    # Load the AlexNet model
    an = load_alexnet().to(device)
    extractor = ActivationExtractor(an, [args['target_layer']])
    pbar = tqdm(enumerate(dataloader))

    for i, (inputs, labels) in pbar:

        inputs = inputs.to(device)
        with torch.no_grad():
            _ = an.features(inputs)
        
        activations = extractor.get_activations()[args['target_layer']].to(device)

        x_recon, z = csae.forward(activations)

        l0_sparsity = torch.sum(z != 0).item() / z.numel()
        l1_sparsity = torch.sum(torch.abs(z)).item() / z.numel()

        l0_sparsity_list.append(l0_sparsity.detach().cpu())
        l1_sparsity_list.append(l1_sparsity.detach().cpu())
        
        extractor.clear_activations()

    return l0_sparsity_list, l1_sparsity_list



        

