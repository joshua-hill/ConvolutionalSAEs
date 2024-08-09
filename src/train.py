import torch
from torch import nn as nn
from torch.nn import functional as F
from csae import CSAE
from tqdm import tqdm
import os
from extract import load_alexnet, ActivationExtractor, create_image_dataloader
from curriculum import Curriculum
import yaml
import argparse
import wandb



def loss_function(x_recon, x, z, decoder_tensor, alpha=1.0):
    #print(f'size of decoder: {decoder_tensor.shape}')
    #print(f'size of z: {z.shape}')
    #print(f'size of decoder_norm: {(torch.norm(decoder_tensor, dim=0, keepdim=True).expand_as(z)).shape}')
    return F.mse_loss(x_recon, x) + alpha * (((torch.norm(decoder_tensor, dim=0, keepdim=True).expand_as(z)) * torch.abs(z)).mean())


def train_step_csae(model, activations, optimizer, loss_func, sparsity_penalty):
    optimizer.zero_grad()
    x_recon, hidden_state = model.forward(activations)
    loss = loss_func(x_recon, activations, hidden_state, model.decoder.weight, sparsity_penalty)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), x_recon.detach()


def train_csae(model, args, wandb_run=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataloader = create_image_dataloader(batch_size=512)
    total_iterations = len(dataloader)

    curriculum = Curriculum(args['training']['curriculum'], total_iterations)

    optimizer = torch.optim.Adam(model.parameters(), lr=curriculum.learning_rate)  
    
    

    state_path = os.path.join(args['out_dir'], "state.pt")

    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for i in range(state["train_step"] + 1):
            curriculum.update()

    alexnet = load_alexnet().to(device)
    extractor = ActivationExtractor(alexnet, [args['target_layer']])

    
    pbar = tqdm(enumerate(dataloader))

    for i, (inputs, _) in pbar:

        inputs = inputs.to(device)
        with torch.no_grad():
            _ = alexnet.features(inputs)
        
        activations = extractor.get_activations()[args['target_layer']]
        loss, output = train_step_csae(model, activations, optimizer, loss_function, curriculum.sparsity_penalty)
        #need to clear activations bc limited memory, have to compute during training because limtied diskspace
        extractor.clear_activations()
        
        if wandb_run and i % args['wandb']['log_every_steps'] == 0:
            wandb_run.log({
                "overall_loss": loss,
                "learning_rate": curriculum.learning_rate,
                "sparsity_penalty": curriculum.sparsity_penalty,
            }, step=i)

        curriculum.update()

        pbar.set_description(f"loss: {loss}, Iteration: {i}/{total_iterations}")

        if i % args['training']['save_every_step'] == 0 and not args['training']['test_run']:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args['training']['keep_every_steps'] > 0
            and i % args['training']['keep_every_steps'] == 0
            and not args['training']['test_run']
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args['out_dir'], f"model_{i}.pt"))
    #save after final iteration
    torch.save(model.state_dict(), os.path.join(args['out_dir'], f"model_{i}.pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    
    cmd_args = parser.parse_args()

    with open(cmd_args.config, 'r') as f:
        config = yaml.safe_load(f)

    wandb.login(key='a57325dff13478a3c53d373211005c6d56047140')
    
    wandb.init(project="C-SAE", config=config, name='C-SAE_Feature_2_Test')

    target_layers = [
        'features.2',  
        'features.5',  
        'features.7',  
        'features.9',  
        'features.12'
    ]

    assert config['target_layer'] in target_layers, "Invalid target layer"

    #size of activations
    target_sizes = {
        'features.2': (64, 27, 27),
        'features.5': (192, 13, 13),
        'features.7': (384, 13, 13),
        'features.9': (256, 13, 13),
        'features.12': (256, 6, 6),
    }

    

    model = CSAE(
        in_channels=target_sizes[config['target_layer']][0],
        dict_size=target_sizes[config['target_layer']][0] * config['dict_factor'],
    )

    # Ensure the output directory exists
    os.makedirs(config['out_dir'], exist_ok=True)
    with open(config['out_dir'] + "/config.yaml", "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    train_csae(model, config, wandb_run=wandb)

