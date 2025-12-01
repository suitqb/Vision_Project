import torch
import torch.nn as nn

class DDPM(nn.Module):
    def __init__(self, model, image_size=64, channels=3,
                 timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__()
        self.model = model
        self.device = device
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise

    def p_losses(self, x0):
        b = x0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0.device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_t, t)
        return torch.mean((noise - noise_pred) ** 2)

    @torch.no_grad()
    def p_sample(self, x, t):
        betas = self.betas
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod
        alphas_cumprod = self.alphas_cumprod

        beta_t = betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_t = sqrt_one_minus[t].view(-1, 1, 1, 1)
        alpha_cum_t = alphas_cumprod[t].view(-1, 1, 1, 1)

        eps_theta = self.model(x, t)
        mean = (1 / torch.sqrt(1 - beta_t)) * (
            x - beta_t / sqrt_one_minus_t * eps_theta
        )

        if (t == 0).all():
            return mean
        noise = torch.randn_like(x)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, batch_size):
        x = torch.randn(batch_size, self.channels, self.image_size,
                        self.image_size, device=self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t] * batch_size, device=self.device).long()
            x = self.p_sample(x, t_batch)
        return x
