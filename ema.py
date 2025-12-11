import torch

class EMA:
    def __init__(self, model, beta=0.9999):
        self.beta = beta
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.beta).add_(p.data, alpha=1 - self.beta)

    @torch.no_grad()
    def copy_to(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name])
