import torch
import torch.nn as nn

#metrics
mse = nn.MSELoss(reduction='none')
def n2s_loss(pred, target, mask):
    """MSE only on blinded (mask==0) pixels."""
    loss_pix = mse(pred, target)
    return (loss_pix * (1 - mask)).sum() / (1 - mask).sum()

@torch.no_grad()
def masked_psnr(pred, target, mask, max_val=1.0):
    """PSNR computed only over masked (held‑out) pixels."""
    mse_val = ((pred - target) ** 2 * (1 - mask)).sum() / (1 - mask).sum()
    return 20 * torch.log10(max_val / torch.sqrt(mse_val))



from skimage.restoration import estimate_sigma # pip install scikit-image>=0.19

def estimate_sigma_batch(batch):
    """
    Wavelet MAD estimator (Donoho & Johnstone, 1994) applied per image.
    Returns a tensor [B] of sigma_hat values.
    """
    sigmas = []
    for img in batch: # img [1,H,W] on CPU, 0‑1
        sig = estimate_sigma(img.squeeze(0).numpy(), channel_axis=None)
        sigmas.append(sig)
    return torch.tensor(sigmas, dtype=batch.dtype)

def debiased_psnr(pred, noisy, mask, sigma_hat, eps=1e-8, max_val=1.0):
    """PSNR estimator unbiased w.r.t. the (unknown) clean target."""
    mse_noisy = ((pred - noisy) ** 2 * (1 - mask)).sum(dim=[1,2,3]) / (1 - mask).sum(dim=[1,2,3])
    mse_clean = torch.clamp(mse_noisy - sigma_hat**2, min=eps)
    psnr = 10 * torch.log10(max_val**2 / mse_clean)
    return psnr.mean().item()



