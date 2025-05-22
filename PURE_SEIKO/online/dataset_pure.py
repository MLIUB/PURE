import os, pdb
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"       # in case you are using a multi GPU workstation, choose your GPU here
from tqdm import tqdm

import sys
import os
from accelerate.utils import broadcast
import copy
cwd = os.getcwd()
sys.path.append(cwd)

import torch.nn as nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import wandb
import numpy as np
import argparse
from aesthetic_scorer import MLPDiff
import datetime
from reward_aesthetic.train_bootstrap import BootstrappedNetwork
import ml_collections

from importlib import resources
ASSETS_PATH = resources.files("assets")


class D_explored(torch.nn.Module):
    def __init__(self, config=None, device=None):
        super().__init__()
        
        self.device = device
        self.x = torch.empty(0, device=self.device)
        self.y = torch.empty(0, device=self.device)
        self.mid_observations = torch.empty(0, device=self.device)
        self.mid_labels = torch.empty(0, device=self.device)
        
        self.noise = 0.1
        
        if config.train.optimism in ['none', 'UCB']:
            self.model = MLPDiff()
        elif config.train.optimism == 'bootstrap':
            self.model = BootstrappedNetwork(input_dim=768, num_heads=4)
        else:
            raise NotImplementedError
        
        self.labeler = MLPDiff()
        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"))
        self.labeler.load_state_dict(state_dict)
        self.labeler.eval()
        self.labeler.requires_grad_(False)
    
    def update(self, new_x, new_y=None, lamda=2):
        # Ensure new_x is on the same device as self.x, as of [num_timesteps, batch_size, features]
        device = self.x.device if self.x.numel() > 0 else new_x.device
        new_x = new_x.to(device)
        
        # Debug shapes
        print(f"new_x shape: {new_x.shape}")
        
        # Number of possible timesteps
        num_timesteps = new_x.shape[0]
        batch_size = new_x.shape[1]
        
        # Create the probability distribution
        weights = torch.tensor([lamda**i for i in range(num_timesteps)], device=device)
        probs = weights / weights.sum()
        
        print(f"Probability distribution: min={probs.min()}, max={probs.max()}, sum={probs.sum()}")
        
        # Simply sample the last timestep for each batch item
        sampled_x = []
        sampled_y = []
        xT = new_x[-1]
        
        with torch.no_grad():
            self.labeler.to(device)
            y = self.labeler(xT)
            for i in range(batch_size):
                # Sample a single index using the probability distribution
                try:
                    idx = torch.multinomial(probs, 1)[0].item()
                    sample = new_x[idx, i]
                except Exception as e:
                    print(f"Error sampling: {e}. Using last timestep instead.")
                    sample = new_x[-1, i]  # Use last timestep as fallback
                    
                # Generate label with noise
                output = self.labeler(sample) + torch.randn_like(sample) * self.noise
                sampled_x.append(sample)
                sampled_y.append(output)
        
        # Convert lists to tensors
        sampled_x = torch.stack(sampled_x, dim=0)
        sampled_y = torch.stack(sampled_y, dim=0)
        
        # Update dataset
        if self.x.numel() == 0:
            self.x = xT
            self.y = y
            self.mid_observations = sampled_x
            self.mid_labels = sampled_y
        else:
            self.x = torch.cat((self.x, xT), dim=0)
            self.y = torch.cat((self.y, y), dim=0)
            self.mid_observations = torch.cat((self.mid_observations, sampled_x), dim=0)
            self.mid_labels = torch.cat((self.mid_labels, sampled_y), dim=0)
        
        print(f"self.x.shape: {self.x.shape}")
        print(f"self.y.shape: {self.y.shape}")
        print(f"self.mid_observations.shape: {self.mid_observations.shape}")
        print(f"self.mid_labels.shape: {self.mid_labels.shape}")
        assert self.x.shape[0] == self.y.shape[0], "Mismatch in samples between self.x and self.y"
        assert self.mid_observations.shape[0] == self.mid_labels.shape[0], "Mismatch in samples between sampled labels"
    
    def cov(self, config=None): # used if we have non-optimism or UCB optimism
        with torch.no_grad():
            if config is not None and config.train.num_gpus > 1:
                features = self.model.module.forward_up_to_second_last(self.mid_observations)
            else:
                features = self.model.forward_up_to_second_last(self.mid_observations)
        return torch.cov(features.t())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


    def train_MLP(self, accelerator, config):
        
        assert self.mid_observations.requires_grad == False
        assert self.mid_labels.requires_grad == False
        
        args = ml_collections.ConfigDict()

        # Arguments
        args.num_epochs = 300
        args.train_bs = 512
        args.val_bs = 512
        args.lr = 0.001
        
        if 'SGLD' in config.train.keys():
            args.SGLD_base_noise = config.train.SGLD
            assert config.train.optimism == 'none', "SGLD only works with non-optimism"
        else:
            args.SGLD_base_noise = 0
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        optimizer = accelerator.prepare(optimizer)
        
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()

        self.model.requires_grad_(True)
        self.model.train()
        
        val_percentage = 0.05 # 5% of the training data will be used for validation
        train_border = int(self.mid_observations.shape[0] * (1 - val_percentage) )
        
        train_dataset = TensorDataset(self.mid_observations[:train_border],self.mid_labels[:train_border])
        train_loader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True) # create your dataloader

        val_dataset = TensorDataset(self.mid_observations[train_border:],self.mid_labels[train_border:])
        val_loader = DataLoader(val_dataset, batch_size=args.val_bs) # create your dataloader
        
        best_loss = 999
        best_model = {k: torch.empty_like(v) for k, v in self.model.state_dict().items()}
            
        def adjust_noise(learning_rate, batch_size):
            return args.SGLD_base_noise * (learning_rate ** 0.5) / (batch_size ** 0.5)   
    
        with torch.enable_grad():
            for epoch in range(args.num_epochs):
                
                noise_level = adjust_noise(args.lr, args.train_bs)
                
                losses = []
                for batch_num, (x,y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = self.model(x)
                    
                    loss = criterion(output, y.detach())
                    accelerator.backward(loss)
                    losses.append(loss.item())
                    
                    # add Gaussian noise to gradients
                    if config.train.num_gpus > 1:
                        for param in self.model.module.parameters():
                            if param.grad is not None:
                                param.grad += noise_level * torch.randn_like(param.grad)
                    else:
                        for param in self.model.parameters():
                            if param.grad is not None:
                                param.grad += noise_level * torch.randn_like(param.grad)

                    optimizer.step()
                
                if accelerator.is_main_process:
                    losses_val = []
                    
                    for _, (x,y) in enumerate(val_loader):
                        self.model.eval()
                        output = self.model(x)
                        loss = criterion2(output, y.detach())

                        losses_val.append(loss.item())

                    print('Epoch %d | Loss %6.4f | val-loss %6.4f' % (epoch, (sum(losses)/len(losses)), sum(losses_val)/len(losses_val)))

                    if sum(losses_val)/len(losses_val) < best_loss:
                        best_loss = sum(losses_val)/len(losses_val)
                        print("Best MAE val loss so far: %6.4f" % (best_loss))
                        best_model = self.model.state_dict()
        
        best_model = broadcast(best_model)
        self.model.load_state_dict(best_model)
             
        self.model.requires_grad_(False)
        self.model.eval()
            
        del optimizer, criterion, criterion2, train_dataset, train_loader, val_dataset, val_loader
  
    def train_bootstrap(self,accelerator,config, lamda=2):
        from reward_aesthetic.train_bootstrap import bootstrapping, BootstrappedDataset
        
        assert self.mid_observations.requires_grad == False
        assert self.mid_labels.requires_grad == False
        
        args = ml_collections.ConfigDict()

        # # Add arguments
        args.num_epochs = 300
        args.train_bs = 512
        args.val_bs = 512
        args.lr = 0.001
        args.num_heads = 4
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        optimizer = accelerator.prepare(optimizer)
        
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()

        self.model.requires_grad_(True)
        self.model.train()
        
        val_percentage = 0.05 # 5% of the training data will be used for validation
        train_border = int(self.mid_observations.shape[0] * (1 - val_percentage))

        train_dataset = TensorDataset(self.mid_observations[:train_border],self.mid_labels[:train_border])
        bootstrapped_traindata = bootstrapping(train_dataset, n_datasets=args.num_heads)
        bootstrapped_trainset = BootstrappedDataset(bootstrapped_traindata)
        train_loader = DataLoader(bootstrapped_trainset, batch_size=args.train_bs, shuffle=True)  
        
        val_dataset = TensorDataset(self.mid_observations[train_border:],self.mid_labels[train_border:])
        bootstrapped_valdata = bootstrapping(val_dataset, n_datasets=args.num_heads)
        bootstrapped_valset = BootstrappedDataset(bootstrapped_valdata)
        val_loader = DataLoader(bootstrapped_valset, batch_size=args.val_bs,shuffle=False)
        
        best_loss = 999
        best_model = {k: torch.empty_like(v) for k, v in self.model.state_dict().items()}
        
        with torch.enable_grad():
            for epoch in range(args.num_epochs):
                
                losses = []
                for _, (inputs,targets) in enumerate(train_loader):
                    
                    optimizer.zero_grad()
                    loss = 0
                    for i in range(args.num_heads):
                        output = self.model(inputs, head_idx=i)
                        loss += criterion(output, targets[:,i,:].detach())
                        
                    loss /= args.num_heads
                    accelerator.backward(loss)
                    losses.append(loss.item())
                    
                    optimizer.step()
                
                if accelerator.is_main_process:
                    losses_val = []    
                    for _, (inputs,targets) in enumerate(val_loader):
                        self.model.eval()
                        optimizer.zero_grad()
                        output = None
                        loss = 0

                        for i in range(args.num_heads):
                            output = self.model(inputs, head_idx=i)
                            loss += criterion(output, targets[:,i,:].detach())
                            
                        loss /= args.num_heads
                        losses_val.append(loss.item())
                                       

                    print('Epoch %d | Loss %6.4f | val-loss %6.4f' % (epoch, (sum(losses)/len(losses)), sum(losses_val)/len(losses_val)))
                    
                    if sum(losses_val)/len(losses_val) < best_loss:
                        best_loss = sum(losses_val)/len(losses_val)
                        print("Best MAE val loss so far: %6.4f" % (best_loss))
                        best_model = self.model.state_dict()

        best_model = broadcast(best_model)
        self.model.load_state_dict(best_model)
             
        self.model.requires_grad_(False)
        self.model.eval()
            
        del optimizer, criterion, criterion2, train_dataset, train_loader, bootstrapped_traindata, bootstrapped_trainset,
        val_dataset, val_loader, bootstrapped_valdata, bootstrapped_valset 

if __name__ == "__main__":
    from accelerate import Accelerator
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = ml_collections.ConfigDict()
    config.train = train = ml_collections.ConfigDict()
    config.train.optimism = 'bootstrap'
    
    dataset = D_explored(config).to(device, dtype=torch.float32)
    
    new_data_x = torch.from_numpy(np.load("./reward_aesthetic/data/ava_x_openclip_l14.npy"))[:200000,:].to(device)
    dataset.update(new_data_x)
    assert len(dataset.x) == len(dataset.y)
    
    accelerator = Accelerator()
    dataset.model = accelerator.prepare(dataset.model)
    
    for name, param in dataset.model.named_parameters():
        print(name)
    
    # print(dataset.cov().shape)
    # print(dataset.cov().min())
    # print(dataset.cov().max())
    # dataset.train_MLP(accelerator,config)
    
    # dataset.train_bootstrap(accelerator,config)
        
