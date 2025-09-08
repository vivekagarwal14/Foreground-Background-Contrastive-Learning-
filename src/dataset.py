import torch
import random
from torch.utils.data import Dataset




class GlomeruliContrastiveDataset(Dataset):
    
    def __init__(self, data, targets, is_training=True):
        """
        Initialize the dataset with data and targets.
        data: All images in the dataset.
        targets: Odorant labels for each image.
        """
        self.data = data
        self.targets = targets  # Odorant labels (0-9 for 10 odorants)
        self.is_training = is_training
        self.generate_indices()  # Generate indices grouped by odorant

        # Set images_per_odorant based on whether this is training or test dataset
        self.images_per_odorant = 500 if is_training else 100

    def generate_indices(self):
        """
        Generate indices for each odorant to allow sampling within and between odorants.
        This helps in efficiently sampling anchor, positive, and negative examples.
        """
        self.foreground_indices = {}  # Store indices for each odorant
        for i, label in enumerate(self.targets):
            label = label.item()  # Get the label for this image (odorant ID)
            if label not in self.foreground_indices:
                self.foreground_indices[label] = []
            self.foreground_indices[label].append(i)

    def __getitem__(self, index):
        """
        Retrieve an anchor, positive, and negative sample for both foreground and background tasks.
        index: The anchor index from which to generate samples.
        Returns:
        anchor_f_img, positive_f_img, negative_f_img: For foreground (odorant-based)
        anchor_b_img, positive_b_img, negative_b_img: For background (mouse-based)
        """
        # --- Foreground Contrastive Learning (Odorant-Based) ---
        # Anchor for Foreground
        anchor_f_img = self.data[index]
        anchor_f_label = self.targets[index].item()

        # Positive for Foreground (same odorant, different image)
        positive_f_idx = index
        while positive_f_idx == index:
            positive_f_idx = random.choice(self.foreground_indices[anchor_f_label])
        positive_f_img = self.data[positive_f_idx]

        # Negative for Foreground (different odorant)
        negative_f_label = anchor_f_label
        while negative_f_label == anchor_f_label:
            negative_f_label = random.randint(0, len(self.foreground_indices) - 1)
        negative_f_idx = random.choice(self.foreground_indices[negative_f_label])
        negative_f_img = self.data[negative_f_idx]

        # --- Background Contrastive Learning (Mouse-Based) ---
        # Anchor for Background is the same as Foreground
        anchor_b_img = anchor_f_img

        # Positive for Background (different odorant, same mouse)
        # Define positive_b_label as a different odorant
        positive_b_label = anchor_f_label
        while positive_b_label == anchor_f_label:
            positive_b_label = random.randint(0, len(self.foreground_indices) - 1)
    
        # Calculate base index and relative index for positive background sample
        base_index = positive_b_label * self.images_per_odorant
        relative_index = index % self.images_per_odorant
        final_index = min(base_index + relative_index, len(self.data) - 1)
        positive_b_img = self.data[final_index]

        # Negative for Background (different mouse, different odorant)
        while True:
            negative_b_idx = random.randint(0, len(self.data) - 1)
            if negative_b_idx != negative_f_idx:
                break
        negative_b_img = self.data[negative_b_idx]

        # Return additional indices for training, or simpler structure for testing
        if self.is_training:
            return (anchor_f_img, positive_f_img, negative_f_img, anchor_b_img, positive_b_img, negative_b_img, 
                    anchor_f_label, index, positive_f_idx, negative_f_idx, final_index, negative_b_idx)
        else:
            return (anchor_f_img, positive_f_img, negative_f_img, anchor_b_img, positive_b_img, negative_b_img, anchor_f_label)

        
    def __len__(self):
        return len(self.data)


#dataset class 
def generate_mask(h, w, p_drop=0.10, block=1):
    """
    Binary mask with 1 = keep, 0 = blind.  Blocks of size `block×block` drop together.
    """
    mask = torch.ones((h, w), dtype=torch.float32)
    anchors = torch.rand((h // block, w // block)) < p_drop
    for y in range(h):
        for x in range(w):
            if anchors[y // block, x // block]:
                mask[y, x] = 0.0
    return mask

class GlomeruliN2SDataset(Dataset):
    def __init__(self, images, labels=None, p_drop=0.10):
        self.images  = images            # Tensor [N,1,H,W]
        self.labels  = labels            # unused for training; kept for any analysis
        self.p_drop  = p_drop
        self.H, self.W = images.shape[-2:]

    def __len__(self): return self.images.shape[0]

    def __getitem__(self, idx):
        img   = self.images[idx]                 # tensor, already 0‑1
        mask  = generate_mask(self.H, self.W, self.p_drop)
        masked_img = img.clone()
        masked_img[0] *= mask                    # channel 0 only

        # return as 3‑tuple needed for N2S loss
        return masked_img, img, mask.unsqueeze(0) # add channel dim to mask





