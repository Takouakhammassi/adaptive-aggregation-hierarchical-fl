import torch


def update_norm(new_w, base_w):
    sq = 0.0
    for k in new_w:
        sq += (new_w[k].float() - base_w[k].float()).pow(2).sum().item()
    return sq ** 0.5


def clip_update(new_w, base_w, clip):
    norm = update_norm(new_w, base_w)
    scale = min(1.0, clip / (norm + 1e-12))
    clipped = {k: base_w[k].float() + (new_w[k].float() - base_w[k].float()) * scale
               for k in new_w}
    return clipped, norm


def add_noise(weights, std, generator):
    return {k: v.float() + torch.randn(v.shape, generator=generator) * std
            for k, v in weights.items()}