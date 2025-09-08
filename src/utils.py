import matplotlib.pyplot as plt


def calculate_dataset_stats(data):
    """
    Calculates statistics (mean, standard deviation, min, max) for a given dataset.

    Args:
        data (torch.Tensor): Dataset tensor in the shape [N, C, H, W].

    Returns:
        dict: A dictionary containing mean, standard deviation, min, and max values.
    """
    # Flatten the data along spatial dimensions
    flat_data = data.view(data.size(0), -1)

    # Calculate statistics
    stats = {
        "mean": flat_data.mean().item(),
        "std_dev": flat_data.std().item(),
        "min_val": flat_data.min().item(),
        "max_val": flat_data.max().item()
    }

    return stats


def dataset_sanity_check(dataset, sample_index=15):
    """
    Perform sanity check on contrastive dataset to ensure indices are correct.
    
    Args:
        dataset: GlomeruliContrastiveDataset instance
        sample_index (int): Index to test (default: 15)
    """
    # Retrieve a single sample to run the sanity check
    (anchor_f_img, positive_f_img, negative_f_img,
     anchor_b_img, positive_b_img, negative_b_img,
     anchor_f_label, anchor_index, positive_f_index, negative_f_index, positive_b_index, negative_b_index) = dataset[sample_index]
    
    print("Sanity Check - Foreground and Background Indices")
    print(f"Foreground - Anchor Index: {anchor_index}")
    print(f"Foreground - Positive Index: {positive_f_index}")
    print(f"Foreground - Negative Index: {negative_f_index}")
    print(f"Background - Anchor Index: {anchor_index}")
    print(f"Background - Positive Index: {positive_b_index}")
    print(f"Background - Negative Index: {negative_b_index}")
    
    # Check if indices are unique as expected
    error_found = False
    
    if anchor_index == positive_f_index:
        print("Error: Foreground anchor and positive are the same!")
        error_found = True
    if anchor_index == negative_f_index:
        print("Error: Foreground anchor and negative are the same!")
        error_found = True
    if anchor_index == positive_b_index:
        print("Error: Background anchor and positive are the same!")
        error_found = True
    if anchor_index == negative_b_index:
        print("Error: Background anchor and negative are the same!")
        error_found = True
    if positive_f_index == negative_f_index:
        print("Error: Foreground positive and negative are the same!")
        error_found = True
    if positive_b_index == negative_b_index:
        print("Error: Background positive and negative are the same!")
        error_found = True
    
    if not error_found:
        print("✅ Sanity check passed! All indices are unique as expected.")
    
    return not error_found



def prepare_image(img):
    """
    Unnormalize and convert tensor to numpy for display.
    
    Args:
        img (torch.Tensor): Image tensor
        
    Returns:
        numpy.ndarray: Prepared image for display
    """
    img = img * 0.5 + 0.5  # Unnormalize
    return img.squeeze(0).numpy()  # Convert to numpy and remove channel dimension


def visualize_contrastive_samples(dataset, sample_index=15, figsize=(12, 8)):
    """
    Visualize contrastive learning samples (anchor, positive, negative) for both foreground and background.
    
    Args:
        dataset: GlomeruliContrastiveDataset instance
        sample_index (int): Index to visualize (default: 15)
        figsize (tuple): Figure size for the plot
    """
    # Get sample from dataset
    (anchor_f_img, positive_f_img, negative_f_img,
     anchor_b_img, positive_b_img, negative_b_img,
     anchor_f_label, anchor_index, positive_f_index, negative_f_index, positive_b_index, negative_b_index) = dataset[sample_index]
    
    # Prepare images for display
    images = [
        prepare_image(anchor_f_img),
        prepare_image(positive_f_img),
        prepare_image(negative_f_img),
        prepare_image(anchor_b_img),
        prepare_image(positive_b_img),
        prepare_image(negative_b_img)
    ]
    
    titles = [
        "Foreground - Anchor", "Foreground - Positive", "Foreground - Negative",
        "Background - Anchor", "Background - Positive", "Background - Negative"
    ]
    
    # Plot images
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()



def save_checkpoint(epoch, net_F, net_B, optimizer, loss, save_dir="/gpfs/data/rinberglab/vivek/checkpoints___alt___"):
    checkpoint = {
        'epoch': epoch,
        'net_F_state_dict': net_F.state_dict(),
        'net_B_state_dict': net_B.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint_epoch_{epoch}.pth")
    print(f"Checkpoint saved for epoch {epoch}")


    

def plot_denoising_results(net_F, net_B, test_loader, device, num_images=5):
    """
    Plot denoising results showing original, foreground, background, denoised, etc.
    """
    # Get a batch from test loader
    noisy_images, positive_f_img, negative_f_img, anchor_b_img, positive_b_img, negative_b_img, labels = next(iter(test_loader))
    noisy_images = noisy_images.to(device)
    
    # Set models to eval mode
    net_F.eval()
    net_B.eval()
    
    with torch.no_grad():
        f_noisy = net_F(noisy_images)
        b_noisy = net_B(noisy_images)
        denoised_images = f_noisy + b_noisy

    # Create the plot
    fig, axes = plt.subplots(6, num_images, figsize=(15, 8))
    row_titles = ['Original Noisy Images', 'Estimated Foreground', 'Estimated Background',
                  'Denoised Images', 'Difference', 'denoise_sub']

    for ax, row in zip(axes[:, 0], row_titles):
        ax.set_ylabel(row, rotation=90, size='large')

    for idx in range(num_images):
        axes[0, idx].imshow(noisy_images[idx].cpu().squeeze(), cmap='gray')
        axes[0, idx].set_title(f"Label: {labels[idx].item()}")
        axes[0, idx].axis('off')

        axes[1, idx].imshow(f_noisy[idx].cpu().squeeze(), cmap='gray')
        axes[1, idx].axis('off')

        axes[2, idx].imshow(b_noisy[idx].cpu().squeeze(), cmap='gray')
        axes[2, idx].axis('off')

        axes[3, idx].imshow(denoised_images[idx].cpu().squeeze(), cmap='gray')
        axes[3, idx].axis('off')

        difference_image = noisy_images[idx].cpu().squeeze() - denoised_images[idx].cpu().squeeze()
        axes[4, idx].imshow(difference_image, cmap='gray')
        axes[4, idx].axis('off')

        denoised_subtracted_image = noisy_images[idx].cpu().squeeze() - b_noisy[idx].cpu().squeeze()
        axes[5, idx].imshow(denoised_subtracted_image, cmap='gray')
        axes[5, idx].axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_curves(train_losses, test_losses, num_epochs):
    """
    Plot training and test loss curves
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs. Test Loss')
    plt.legend()
    plt.show()


# Function to plot noisy and denoised images for a specific odorant and image index
def plot_specific_odorant_image(odorant_idx, image_idx, net_F, net_B, test_loader, device):
    # Set the networks to evaluation mode
    net_F.eval()
    net_B.eval()

    # Find the batch that contains the desired odorant and image
    for noisy_images, _, _, _, _, _, labels in test_loader:
        # Move the images and labels to the correct device
        noisy_images = noisy_images.to(device)
        labels = labels.to(device)

        # Check if the current batch contains the desired odorant
        if odorant_idx in labels:
            # Get the indices of the desired odorant in this batch
            odorant_indices = (labels == odorant_idx).nonzero(as_tuple=True)[0]

            if image_idx < len(odorant_indices):
                chosen_idx = odorant_indices[image_idx]

                # Get the noisy image
                noisy_image = noisy_images[chosen_idx].unsqueeze(0)

                # Pass the image through the networks
                with torch.no_grad():
                    f_noisy = net_F(noisy_image)
                    b_noisy = net_B(noisy_image)
                    #denoised_image = f_noisy + b_noisy  
                    denoised_subtracted_image = f_noisy

                # Convert images to CPU and numpy for visualization
                noisy_image = noisy_image.squeeze().cpu().numpy()
                denoised_subtracted_image = denoised_subtracted_image.squeeze().cpu().numpy()

                # Plot the noisy and denoised image
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                # Plot noisy image
                axes[0].imshow(noisy_image, cmap='gray')
                axes[0].set_title(f"Noisy Image (Odorant {odorant_idx})")
                axes[0].axis('off')

                # Plot denoised image
                axes[1].imshow(denoised_subtracted_image, cmap='gray')
                axes[1].set_title("Denoised Image")
                axes[1].axis('off')

                plt.show()
                return

    print(f"No image with index {image_idx} found for odorant {odorant_idx}.")

# Example usage
#odorant_idx = 1  #  odorant index 
#image_idx = 1    # index of image within odorant

# Plot the selected noisy and denoised images
#plot_specific_odorant_image(odorant_idx, image_idx, net_F, net_B, test_loader, device)

def get_sorted_checkpoints(checkpoints_dir):
    """
    Get sorted list of checkpoint files by epoch number
    """
    import os
    import re
    
    checkpoints = sorted(
        [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith(".pth")],
        key=lambda x: int(re.search(r'checkpoint_epoch_(\d+).pth', x).group(1))
    )
    return checkpoints


def visualize_image_and_mask(image, bboxes, image_id):
    """
    Create and visualize the mask for an image alongside the original image
    
    Args:
        image (np.array): Grayscale image
        bboxes (list): List of bounding boxes [x, y, w, h]
        image_id (int): ID of the image for labeling
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a mask from the bounding boxes
    image_mask = np.zeros(image.shape)
    for box in bboxes:
        x, y, w, h = map(int, box)
        image_mask[y:y+h, x:x+w] = 1
    
    # Plot the image and its mask side by side
    plt.figure(figsize=(12, 6))
    
    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Image ID {image_id}: Original Image")
    plt.axis('off')
    
    # Plot the mask
    plt.subplot(1, 2, 2)
    plt.imshow(image_mask, cmap='gray')
    plt.title(f"Image ID {image_id}: ROI Mask")
    plt.axis('off')
    
    plt.show()


def load_checkpoint(checkpoint_path, net_F, net_B, device):
    """
    Load a checkpoint and restore model states
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        net_F: Foreground network model
        net_B: Background network model  
        device: Device to load checkpoint on
        
    Returns:
        dict: Checkpoint data including epoch and loss
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    net_F.load_state_dict(checkpoint['net_F_state_dict'])
    net_B.load_state_dict(checkpoint['net_B_state_dict'])
    
    # Set models to evaluation mode
    net_F.eval()
    net_B.eval()
    
    print(f"Checkpoint loaded from {checkpoint_path} (Epoch {checkpoint['epoch']})")
    return checkpoint


def calculate_r_for_checkpoint(checkpoint_path, net_F, net_B, images, device):
    """
    Calculate R metrics for all images using a specific checkpoint
    """
    from src.data_processing import calculate_r_metric_single
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_F.load_state_dict(checkpoint['net_F_state_dict'])
    net_B.load_state_dict(checkpoint['net_B_state_dict'])
    net_F.eval()
    net_B.eval()
    
    odor_r_metrics = {}
    
    for image_id, data in images.items():
        image = data['image']
        bboxes = [ann['bbox'] for ann in data['bboxes']]
        
        input_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            denoised_output = net_F(input_tensor)
        
        denoised_image = denoised_output.squeeze().cpu().numpy()
        r_metric = calculate_r_metric_single(denoised_image, bboxes)
        
        odor_label = image_id - 1
        odor_r_metrics[odor_label] = r_metric
    
    return odor_r_metrics


def plot_r_metric_trends(r_trends, original_r_metrics, epoch_numbers, step=10):
    """
    Plot R metric trends across checkpoints with original baselines
    
    Args:
        r_trends (dict): Dictionary of odor -> list of R values
        original_r_metrics (dict): Dictionary of odor -> original R metric
        epoch_numbers (list): List of epoch numbers for x-axis
        step (int): Step size for x-axis labels
    """
    plt.figure(figsize=(12, 6))
    
    for odor, r_values in r_trends.items():
        color = f"C{odor}"
        
        # Plot R metric trend for the odor
        plt.plot(range(len(r_values)), r_values, label=f"Odor {odor} (Denoised)", color=color)
        
        # Add horizontal line for the original R metric
        original_r_metric = original_r_metrics[odor]
        plt.axhline(y=original_r_metric, color=color, linestyle='--', linewidth=0.8)
        
        # Annotate the horizontal line
        plt.text(len(r_values) - 1, original_r_metric, 
                 f"Odor {odor}: R={original_r_metric:.4f}", 
                 color=color, fontsize=8, verticalalignment='bottom')
    
    # Adjust x-axis ticks
    plt.xticks(ticks=range(0, len(epoch_numbers), step), 
               labels=[epoch_numbers[i] for i in range(0, len(epoch_numbers), step)], 
               rotation=90)
    
    # Styling
    plt.legend(title="Odors", loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.xlabel("Checkpoints (Every 2 Epochs)")
    plt.ylabel("R Metric")
    plt.title("Mean R Metric Trends Across Checkpoints with Original Baseline")
    plt.grid()
    plt.tight_layout()
    plt.show()


def calculate_roi_ratios_for_checkpoint(checkpoint_path, net_F, net_B, images, device):
    """
    Calculate ROI intensity ratios (denoised/raw) for each bounding box at a checkpoint
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        net_F, net_B: Network models
        images (dict): Image data with bboxes
        device: Device for computation
        
    Returns:
        dict: {odor_label: [list_of_bbox_ratios]}
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_F.load_state_dict(checkpoint['net_F_state_dict'])
    net_B.load_state_dict(checkpoint['net_B_state_dict'])
    net_F.eval()
    net_B.eval()
    
    odor_roi_ratios = {}
    
    for image_id, data in images.items():
        raw_image = data['image']
        bboxes = [ann['bbox'] for ann in data['bboxes']]
        
        # Prepare input tensor
        input_tensor = torch.tensor(raw_image, dtype=torch.float32)
        if raw_image.ndim == 2:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        else:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(device)
        
        # Generate denoised output
        with torch.no_grad():
            denoised_output = net_F(input_tensor)
        denoised_image = denoised_output.squeeze().cpu().numpy()
        
        # Calculate ratios for each bounding box
        roi_ratios = []
        for bbox in bboxes:
            x, y, w, h = map(int, bbox)
            
            sum_raw = np.sum(raw_image[y:y+h, x:x+w])
            sum_denoised = np.sum(denoised_image[y:y+h, x:x+w])
            
            ratio = sum_denoised / sum_raw if sum_raw != 0 else float('inf')
            roi_ratios.append(ratio)
        
        odor_label = image_id - 1
        odor_roi_ratios[odor_label] = roi_ratios
    
    return odor_roi_ratios


def plot_mean_roi_ratios(roi_ratio_trends, epoch_numbers, selected_odor=0):
    """
    Plot mean ROI ratios for a selected odor across epochs
    
    Args:
        roi_ratio_trends (dict): Dictionary of odor -> list of ratio lists
        epoch_numbers (list): List of epoch numbers for x-axis
        selected_odor (int): Odor index to plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    mean_ratios = [np.mean(r_list) for r_list in roi_ratio_trends[selected_odor]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, mean_ratios, marker='o', linestyle='-')
    plt.title(f"Mean ROI Ratio for Odor {selected_odor} (Denoised/Raw)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean ROI Ratio")
    plt.grid(True)
    plt.show()