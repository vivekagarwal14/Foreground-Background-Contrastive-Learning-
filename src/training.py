# src/training.py
import torch
from torch.nn.functional import pairwise_distance

def compute_loss(anchor_f, positive_f, negative_f,
                 anchor_b, positive_b, negative_b,
                 net_F, net_B,
                 weight_recon, weight_subject, weight_stimulus, weight_contrast):
    """
    Computes the total loss (stimulus, subject, contrast, reconstruction).
    Returns the scalar loss tensor.
    """
    with torch.set_grad_enabled(net_F.training):
        # Forward passes
        f_anchor = net_F(anchor_f)
        b_anchor = net_B(anchor_b)
        f_positive = net_F(positive_f)
        b_positive = net_B(positive_b)
        f_negative = net_F(negative_f)
        b_negative = net_B(negative_b)

        # Calculate losses
        loss_stimulus = torch.log(
            1 + torch.exp(
                torch.clamp(
                    pairwise_distance(f_anchor, f_positive).pow(2)
                    - pairwise_distance(f_anchor, f_negative).pow(2),
                    min=-50, max=50
                )
            )
        ).mean()

        loss_subject = torch.log(
            1 + torch.exp(
                torch.clamp(
                    pairwise_distance(b_anchor, b_positive).pow(2)
                    - pairwise_distance(b_anchor, b_negative).pow(2),
                    min=-50, max=50
                )
            )
        ).mean()

        loss_contrast = torch.log(
            1 + torch.exp(
                -torch.clamp(
                    pairwise_distance(f_anchor, b_anchor).pow(2),
                    min=-50, max=50
                )
            )
        ).mean()

        loss_recon = ((anchor_f - (f_anchor + b_anchor)) ** 2).mean()

        # Weighted total loss
        loss = (weight_recon    * loss_recon +
                weight_subject  * loss_subject +
                weight_stimulus * loss_stimulus +
                weight_contrast * loss_contrast)

    return loss


def train_epoch(net_F, net_B, train_loader, optimizer, device, weights, print_freq=100):
    """Train for one epoch"""
    net_F.train()
    net_B.train()
    
    running_loss = 0.0
    
    for batch_idx, (anchor_f, positive_f, negative_f,
                    anchor_b, positive_b, negative_b,
                    anchor_f_label, anchor_index, positive_f_index, negative_f_index,
                    positive_b_index, negative_b_index) in enumerate(train_loader):

        # Move tensors to device
        anchor_f = anchor_f.to(device)
        positive_f = positive_f.to(device)
        negative_f = negative_f.to(device)
        anchor_b = anchor_b.to(device)
        positive_b = positive_b.to(device)
        negative_b = negative_b.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Compute loss
        loss = compute_loss(anchor_f, positive_f, negative_f,
                           anchor_b, positive_b, negative_b,
                           net_F, net_B,
                           weights['recon'], weights['subject'],
                           weights['stimulus'], weights['contrast'])

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_F.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(net_B.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        # Print progress
        if batch_idx % print_freq == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%) "
                  f"Loss: {loss.item():.6f}")

    return running_loss / len(train_loader)


def validate_epoch(net_F, net_B, test_loader, device, weights):
    """Validate for one epoch"""
    net_F.eval()
    net_B.eval()
    
    running_loss = 0.0
    
    with torch.no_grad():
        for (anchor_f, positive_f, negative_f,
             anchor_b, positive_b, negative_b,
             labels) in test_loader:

            # Move to device
            anchor_f = anchor_f.to(device)
            positive_f = positive_f.to(device)
            negative_f = negative_f.to(device)
            anchor_b = anchor_b.to(device)
            positive_b = positive_b.to(device)
            negative_b = negative_b.to(device)

            # Compute loss
            loss = compute_loss(anchor_f, positive_f, negative_f,
                               anchor_b, positive_b, negative_b,
                               net_F, net_B,
                               weights['recon'], weights['subject'],
                               weights['stimulus'], weights['contrast'])
            
            running_loss += loss.item()

    return running_loss / len(test_loader)