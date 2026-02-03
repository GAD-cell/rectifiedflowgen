from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
import torch
from torchcfm.utils import NeuralODE
import torch.nn.functional as F

def compute_fid_score(real_loader, gen_images, device):
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    
    for x_real, _ in real_loader:
        x_real_uint8 = ((x_real * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).to(device)
        fid_metric.update(x_real_uint8, real=True)
        
    x_gen_uint8 = ((gen_images * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8).to(device)
    fid_metric.update(x_gen_uint8, real=False)
    
    return fid_metric.compute().item()

def build_memory_bank(loader, model_dino, device):
    bank = []
    transform_dino = transforms.Resize(224) 
    
    with torch.no_grad():
        for x, _ in loader:
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

def generate_images(net_model, num=1000, batch_size=100, steps=10, device="cuda"):
    net_model.eval()
    node = NeuralODE(net_model, solver="euler")
    all_images = []
    
    num_batches = num // batch_size
    
    print(f"Generating {num} images in {num_batches} batches...")
    
    with torch.no_grad():
        for _ in range(num_batches):
            z0 = torch.randn(batch_size, 3, 32, 32).to(device)
            
            t_span = torch.linspace(0, 1, 2).to(device)

            traj = node.trajectory(z0, t_span=t_span, steps=steps)
            z1 = traj[-1]
            all_images.append(z1.cpu())
            
    return torch.cat(all_images, dim=0)