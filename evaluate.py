from torchcfm.models.unet.unet import UNetModelWrapper
from utils import compute_fid_score, compute_fid_score_rec, build_memory_bank, build_memory_bank_rec, generate_images, compute_nn_distance, visualize_memorization
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch 
from utils import ReflowDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
def main(is_rectified=False,path_dataset="reflow_3_dataset.pt"):
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

    if is_rectified:
        dataset = ReflowDataset(torch.load(path_dataset))
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=2)
    else:
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=2)

    net_model.load_state_dict(torch.load("model_2rf.pt", map_location=device))

    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()

    images_generees = generate_images(net_model, num=1000).to(device)

    if is_rectified:
        bank = build_memory_bank_rec(dataloader, dino, device)
        fid = compute_fid_score_rec(dataloader, images_generees, device)
    else:
        bank = build_memory_bank(dataloader, dino, device)
        fid = compute_fid_score(dataloader, images_generees, device)

    similarity = compute_nn_distance(images_generees, bank, dino, device)
    
    visualize_memorization(
        gen_images=images_generees,  
        memory_bank=bank,           
        real_dataset=dataloader, 
        model_dino=dino,           
        device=device,
        num_samples=5                
        )
    print(f"FID: {fid:.2f} | NN Similarity: {similarity:.4f}")

if __name__ == "__main__":
    main()