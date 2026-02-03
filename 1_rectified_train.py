import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchcfm.models.unet.unet import UNetModelWrapper
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    print(f"Training 1-Rectified Flow on {device}...")
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=2)

    optimizer = torch.optim.AdamW(net_model.parameters(), lr=1e-4)
    fm = TargetConditionalFlowMatcher(sigma=0.0)

    num_epochs = 10

    for epoch in range(num_epochs):
        net_model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for x1, y in pbar:
                x1 = x1.to(device)
                x0 = torch.randn_like(x1)
                
                t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
                
                vt = net_model(t, xt)
                loss = torch.mean((vt - ut) ** 2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

    torch.save(net_model.state_dict(), "model_1rf.pt")
    print("Model 1-Rectified saved")

if __name__ == '__main__':
    main()