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


import matplotlib.pyplot as plt

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

