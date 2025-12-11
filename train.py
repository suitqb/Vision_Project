import os
import torch
from torch import optim
from torchvision.utils import save_image

from dataset import get_anime_dataloader
from model_unet import UNet
from diffusion import DDPM
from ema import EMA
from metrics import MetricsTracker

def train(num_epochs=300,
          batch_size=32,
          lr=1e-4,              # LR un peu plus faible
          timesteps=400,
          image_size=64,
          device=None,
          resume=True,
          max_steps=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ---------------------------
    # DataLoader
    # ---------------------------
    dl, _ = get_anime_dataloader(
        image_size=image_size,
        batch_size=batch_size,
        # vérifier que les images sont bien normalisées en [-1, 1]
    )

    # ---------------------------
    # Modèle + diffusion
    # ---------------------------
    model = UNet(img_channels=3).to(device)
    ddpm = DDPM(model, image_size=image_size, channels=3,
                timesteps=timesteps, device=device).to(device)
    ddpm.train()  # s'assurer qu'on est bien en mode train

    optim_ = optim.Adam(ddpm.parameters(), lr=lr)

    # AMP (mixed precision)
    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    metrics_tracker = MetricsTracker(
        log_dir="metrics",
        csv_filename="training_metrics.csv",
        resume=resume,
    )
    log_interval = 50
    plot_interval = 500

    def finalize_metrics():
        metrics_tracker.generate_plots(metrics=("loss",), smoothing=20)

    # ---------------------------
    # Reprise éventuelle
    # ---------------------------
    start_epoch = 0
    step = 0
    last_ckpt = "checkpoints/last.pt"

    if resume and os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device)
        ddpm.load_state_dict(ckpt["model"])
        optim_.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        step = ckpt.get("step", 0)
        if "scaler" in ckpt and use_amp and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"[*] Reprise à epoch {start_epoch}, step {step}")

    ema = EMA(ddpm, beta=0.9999)
    # ---------------------------
    # Boucle d'entraînement
    # ---------------------------
    save_interval = 2000   # sauvegarde toutes les 2000 updates

    for epoch in range(start_epoch, num_epochs):
        for x, _ in dl:
            if max_steps is not None and step >= max_steps:
                print(f"Max steps {max_steps} atteint, arrêt.")
                finalize_metrics()
                return

            x = x.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = ddpm.p_losses(x)

            optim_.zero_grad()
            scaler.scale(loss).backward()

            # stabiliser un peu
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)

            scaler.step(optim_)
            scaler.update()
            # ema.update(ddpm)

            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

            if step % log_interval == 0:
                metrics_tracker.log(
                    epoch=epoch,
                    step=step,
                    loss=float(loss.item()),
                    lr=optim_.param_groups[0]["lr"],
                )

            if step % plot_interval == 0:
                metrics_tracker.generate_plots(metrics=("loss",), smoothing=20)

            if step % save_interval == 0:
                # ----- SAMPLE EN MODE EVAL -----
                ddpm.eval()
                with torch.no_grad():
                    # ema.copy_to(ddpm)
                    samples = ddpm.ddim_sample(batch_size=16, ddim_steps=50)
                    # suppose que le modèle travaille en [-1, 1]
                    samples = (samples.clamp(-1, 1) + 1) / 2
                    save_image(
                        samples,
                        f"samples/epoch{epoch}_step{step}.png",
                        nrow=4
                    )
                ddpm.train()
                # -------------------------------

                state = {
                    "model": ddpm.state_dict(),
                    "optimizer": optim_.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "scaler": scaler.state_dict() if use_amp else None,
                    "config": {
                        "batch_size": batch_size,
                        "lr": lr,
                        "timesteps": timesteps,
                        "image_size": image_size,
                    },
                }
                torch.save(
                    state,
                    f"checkpoints/ddpm_epoch{epoch}_step{step}.pt"
                )
                torch.save(state, last_ckpt)  # checkpoint de reprise

            step += 1

    finalize_metrics()


if __name__ == "__main__":
    # exemple : 100 000 updates max
    train(max_steps=100000)
