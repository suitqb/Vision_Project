import torch
from torchvision.utils import save_image
from model_unet import UNet
from diffusion import DDPM

def generate(checkpoint_path, num_samples=64, image_size=64, timesteps=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(img_channels=3).to(device)
    ddpm = DDPM(model, image_size=image_size, channels=3,
                timesteps=timesteps, device=device).to(device)
    ddpm.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ddpm.eval()

    with torch.no_grad():
        x = ddpm.sample(batch_size=num_samples)
        x = (x.clamp(-1,1) + 1) / 2
        save_image(x, "pokemon_generated.png", nrow=8)

if __name__ == "__main__":
    generate("checkpoints/ddpm_epoch50_step2000.pt")
