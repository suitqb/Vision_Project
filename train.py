import torch
from torch import optim
from torchvision.utils import save_image
from dataset import get_pokemon_dataloader
from model_unet import UNet
from diffusion import DDPM
import os

def train(num_epochs=100, batch_size=64, lr=2e-4,
          timesteps=1000, image_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dl, _ = get_pokemon_dataloader(image_size=image_size,
                                   batch_size=batch_size)
    model = UNet(img_channels=3).to(device)
    ddpm = DDPM(model, image_size=image_size, channels=3,
                timesteps=timesteps, device=device).to(device)

    optim_ = optim.Adam(ddpm.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    step = 0
    for epoch in range(num_epochs):
        for x, _ in dl:
            x = x.to(device)
            loss = ddpm.p_losses(x)
            optim_.zero_grad()
            loss.backward()
            optim_.step()

            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
            if step % 2000 == 0:
                with torch.no_grad():
                    samples = ddpm.sample(batch_size=16)
                    samples = (samples.clamp(-1,1) + 1) / 2
                    save_image(samples, f"samples/epoch{epoch}_step{step}.png",
                               nrow=4)
                torch.save(ddpm.state_dict(),
                           f"checkpoints/ddpm_epoch{epoch}_step{step}.pt")

            step += 1

if __name__ == "__main__":
    train()
