from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
import torch
from torchdyn.core import NeuralODE
import torch.nn.functional as F
from torch.utils.data import Dataset

class ReflowDataset(Dataset):
    def __init__(self, pairs):
        self.data = []
        for z0_batch, z1_batch, y_batch in pairs:
            for i in range(z0_batch.shape[0]):
                self.data.append((z0_batch[i], z1_batch[i], y_batch[i]))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def compute_fid_score(real_loader, gen_images, device):
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    
    for x_real, _ in real_loader:
        x_real_uint8 = ((x_real * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).to(device)
        fid_metric.update(x_real_uint8, real=True)
        
    x_gen_uint8 = ((gen_images * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).to(device)
    fid_metric.update(x_gen_uint8, real=False)
    
    return fid_metric.compute().item()

def compute_fid_score_rec(real_loader, gen_images, device):
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    
    for _,x_real, _ in real_loader:
        x_real_uint8 = ((x_real * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).to(device)
        fid_metric.update(x_real_uint8, real=True)
        
    x_gen_uint8 = ((gen_images * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).to(device)
    fid_metric.update(x_gen_uint8, real=False)
    
    return fid_metric.compute().item()

from tqdm import tqdm

def build_memory_bank(loader, model_dino, device):
    bank = []
    transform_dino = transforms.Resize(224) 
    
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Building Memory Bank", unit="batch"):
            x = x.to(device)
            x_norm = transform_dino(x * 0.5 + 0.5)
            
            feats = model_dino(x_norm)
            feats = F.normalize(feats, dim=1)
            bank.append(feats.cpu())
            
    return torch.cat(bank, dim=0).to(device)

def build_memory_bank_rec(loader, model_dino, device):
    bank = []
    transform_dino = transforms.Resize(224) 
    
    with torch.no_grad():
        for _, x, _ in tqdm(loader, desc="Building Memory Bank", unit="batch"):
            x = x.to(device)
            x_norm = transform_dino(x * 0.5 + 0.5)
            
            feats = model_dino(x_norm)
            feats = F.normalize(feats, dim=1)
            bank.append(feats.cpu())
            
    return torch.cat(bank, dim=0).to(device)

def compute_nn_distance(gen_images, memory_bank, model_dino, device):
    transform_dino = transforms.Resize(224)
    
    with torch.no_grad():
        x_gen_norm = transform_dino(gen_images * 0.5 + 0.5)
        gen_feats = model_dino(x_gen_norm)
        gen_feats = F.normalize(gen_feats, dim=1)
        
        cos_sim = torch.mm(gen_feats, memory_bank.T)
        max_sim, _ = torch.max(cos_sim, dim=1)
        
    return max_sim.mean().item()

def generate_images(net_model, num=1000, batch_size=100, device="cuda"):
    net_model.eval()
    node = NeuralODE(net_model, solver="euler")
    all_images = []
    
    num_batches = num // batch_size
    
    print(f"Generating {num} images in {num_batches} batches...")
    
    with torch.no_grad():
        for _ in range(num_batches):
            z0 = torch.randn(batch_size, 3, 32, 32).to(device)
            
            t_span = torch.linspace(0, 1, 2).to(device)

            traj = node.trajectory(z0, t_span=t_span)
            z1 = traj[-1]
            all_images.append(z1.cpu())
            
    return torch.cat(all_images, dim=0)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

def visualize_memorization(gen_images, memory_bank, real_dataset, model_dino, device, num_samples=5, save_path="memorization_check.png"):
    model_dino.eval()
    transform_dino = transforms.Resize(224)

    actual_num = min(num_samples, len(gen_images))
    indices_gen = torch.randperm(len(gen_images))[:actual_num]
    selected_gen = gen_images[indices_gen].to(device) # [N, 3, 32, 32]
    

    with torch.no_grad():
        x_gen_norm = transform_dino(selected_gen * 0.5 + 0.5) 
        gen_feats = model_dino(x_gen_norm)
        gen_feats = F.normalize(gen_feats, dim=1) # [N, 384]

        sim_matrix = torch.mm(gen_feats, memory_bank.T)
        best_sims, best_indices = torch.max(sim_matrix, dim=1)

    fig, axes = plt.subplots(actual_num, 2, figsize=(6, 3 * actual_num))
    if actual_num == 1: axes = [axes] 
    
    for i in range(actual_num):

        img_gen = selected_gen[i].cpu().permute(1, 2, 0).numpy()
        img_gen = (img_gen * 0.5 + 0.5).clip(0, 1)
        
        ax_gen = axes[i][0] if actual_num > 1 else axes[0]
        ax_gen.imshow(img_gen)
        ax_gen.set_title("Générée (2-RF)")
        ax_gen.axis('off')

        idx_real = best_indices[i].item()
        item = real_dataset[idx_real]

        if isinstance(item, tuple):
            if len(item) == 3: 
                img_real_tensor = item[1] 
            else:
                img_real_tensor = item[0]
        else:
            img_real_tensor = item

        img_real = img_real_tensor.permute(1, 2, 0).numpy()
        img_real = (img_real * 0.5 + 0.5).clip(0, 1)
        
        ax_real = axes[i][1] if actual_num > 1 else axes[1]
        ax_real.imshow(img_real)
        score = best_sims[i].item()
        
        color = 'red' if score > 0.95 else 'green'
        ax_real.set_title(f"Voisin Réel\nSim: {score:.4f}", color=color, fontweight='bold')
        ax_real.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Comparaison saved : {save_path}")
    plt.close()