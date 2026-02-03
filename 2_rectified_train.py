import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchcfm.models.unet.unet import UNetModelWrapper
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from torchcfm.utils import NeuralODE

device = "cuda" if torch.cuda.is_available() else "cpu"

net_model = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=128,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)

net_model.load_state_dict(torch.load("model_1rf.pt", map_location=device))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
origin_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
origin_loader = DataLoader(origin_dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=2)

node = NeuralODE(net_model, solver="euler")
reflow_pairs = []

net_model.eval()
print("Generating dataset (Z0 -> Z1)...")

with torch.no_grad():
    for _, y in origin_loader:
        y = y.to(device)
        
        z0 = torch.randn(y.shape[0], 3, 32, 32).to(device)
        
        traj = node.trajectory(z0, t_span=torch.linspace(0, 1, 2).to(device), context={"y": y})
        z1 = traj[-1]
        
        reflow_pairs.append((z0.cpu(), z1.cpu(), y.cpu()))

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

reflow_dataset = ReflowDataset(reflow_pairs)
reflow_loader = DataLoader(reflow_dataset, batch_size=128, shuffle=True, drop_last=True)

optimizer = torch.optim.AdamW(net_model.parameters(), lr=5e-5)
fm = TargetConditionalFlowMatcher(sigma=0.0)

num_epochs_reflow = 50

print("Training 2-Rectified Flow...")

for epoch in range(num_epochs_reflow):
    net_model.train()
    total_loss = 0
    
    for z0, z1, y in reflow_loader:
        z0 = z0.to(device)
        z1 = z1.to(device)
        y = y.to(device)
        
        t, xt, ut = fm.sample_location_and_conditional_flow(z0, z1)
        
        vt = net_model(t, xt)
        loss = torch.mean((vt - ut) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"2-RF Epoch {epoch+1} | Loss: {total_loss / len(reflow_loader):.4f}")

torch.save(net_model.state_dict(), "model_2rf.pt")
print("2-Rectified Flow saved.")