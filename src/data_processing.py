import torch
import json
import os

def load_glomeruli_data(train_path, test_path, normalize_offset=0.5):
    """
    Load glomeruli training and test data from .pt files
    
    Args:
        train_path (str): Path to training data .pt file
        test_path (str): Path to test data .pt file
        normalize_offset (float): Offset to add for normalization (default: 0.5)
    
    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """
    
    # Load training data
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at: {train_path}")
    
    train_data, train_labels = torch.load(train_path)
    print(f"Loaded training data: {train_data.shape}")
    
    # Load test data  
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at: {test_path}")
        
    test_data, test_labels = torch.load(test_path)
    print(f"Loaded test data: {test_data.shape}")
    
    # Apply normalization offset
    train_data = train_data + normalize_offset
    test_data = test_data + normalize_offset
    
    print(f"Applied normalization offset: +{normalize_offset}")
    
    return train_data, train_labels, test_data, test_labels


def load_coco_annotations(data_path, annotations_filename="raw_images_annotations.json"):
    """
    Load COCO format annotations and image metadata
    
    Args:
        data_path (str): Path to the dataset directory
        annotations_filename (str): Name of the annotations JSON file
        
    Returns:
        tuple: (image_info, annotations_info) from COCO data
    """
    annotations_file = os.path.join(data_path, annotations_filename)
    
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Annotations file not found at: {annotations_file}")
    
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)
    
    image_info = coco_data['images']
    annotations_info = coco_data['annotations']
    
    print(f"Loaded {len(image_info)} images and {len(annotations_info)} annotations")
    print("Sample Image Info:", image_info[0])
    print("Sample Annotation Info:", annotations_info[0])
    
    return image_info, annotations_info


def create_image_annotation_mapping(annotations_info):
    """
    Create a mapping from image_id to its annotations
    
    Args:
        annotations_info (list): List of annotation dictionaries from COCO data
        
    Returns:
        dict: Mapping from image_id to list of annotations
    """
    image_id_to_annotations = {}
    for annotation in annotations_info:
        image_id = annotation['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(annotation)
    
    print(f"Created mapping for {len(image_id_to_annotations)} images")
    return image_id_to_annotations


import cv2

def load_images_with_annotations(image_info, image_id_to_annotations, data_path):
    """
    Load images from file paths and combine with their annotations
    
    Args:
        image_info (list): Image metadata from COCO data
        image_id_to_annotations (dict): Mapping from image_id to annotations
        data_path (str): Base path to image files
        
    Returns:
        dict: Dictionary mapping image_id to {'image': array, 'bboxes': list}
    """
    images = {}
    
    for img in image_info:
        image_id = img['id']
        file_name = img['file_name']
        img_path = os.path.join(data_path, file_name)
        
        # Read the image as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = (image / 255.0 - 0.5) * 2
        image = image + 0.5
        
        # Store the image and its corresponding bounding boxes
        images[image_id] = {
            'image': image,
            'bboxes': image_id_to_annotations.get(image_id, [])
        }
    
    print(f"Loaded {len(images)} images with annotations.")
    return images


import numpy as np

def calculate_roi_pixel_sum(image, bboxes):
    """
    Calculate pixel sum within regions of interest (ROIs) defined by bounding boxes
    
    Args:
        image (np.array): Grayscale image
        bboxes (list): List of bounding boxes [x, y, w, h]
        
    Returns:
        float: Sum of pixel values within all ROI regions
    """
    image_mask = np.zeros(image.shape)
    for box in bboxes:
        x, y, w, h = map(int, box)  # Convert bounding box coordinates to integers
        image_mask[y:y+h, x:x+w] = 1  # Create a mask for the bounding box
    return np.sum(image * image_mask)  # Element-wise multiplication and sum


def calculate_r_metrics(images):
    """
    Calculate R metric (ROI pixel sum / total pixel sum) for all images
    
    Args:
        images (dict): Dictionary of image data with bboxes
        
    Returns:
        dict: Dictionary mapping image_id to R metric value
    """
    r_metrics = {}
    
    for image_id, data in images.items():
        image = data['image']
        bboxes = [ann['bbox'] for ann in data['bboxes']]
        
        # Calculate total pixel sum
        total_pixel_sum = np.sum(image)
        
        # Calculate total ROI pixel sum
        total_roi_pixel_sum = calculate_roi_pixel_sum(image, bboxes)
        
        # Calculate R metric
        R_metric = total_roi_pixel_sum / total_pixel_sum if total_pixel_sum > 0 else 0
        r_metrics[image_id] = R_metric
        
        # Print results
        print(f"Image ID {image_id}:")
        print(f"  Total Pixel Sum = {total_pixel_sum}")
        print(f"  Total ROI Pixel Sum = {total_roi_pixel_sum}")
        print(f"  R Metric = {R_metric:.4f}")
    
    return r_metrics


def create_roi_mask(image_shape, bboxes):
    """
    Creates a binary mask with ROIs set to 1 based on bounding boxes.
    """
    mask = np.zeros(image_shape, dtype=np.float32)
    for box in bboxes:
        x, y, w, h = map(int, box)
        mask[y:y+h, x:x+w] = 1
    return mask

def calculate_r_metric_single(image, bboxes):
    """
    Calculate R metric for a single image
    """
    roi_mask = create_roi_mask(image.shape, bboxes)
    roi_pixel_sum = np.sum(image * roi_mask)
    total_pixel_sum = np.sum(image)
    return roi_pixel_sum / total_pixel_sum if total_pixel_sum > 0 else 0


# read .pt for baseline methods comparison
def load_pt_bundle(path):
    """
    Loads .pt file and returns two tensors:
    images : Tensor [N, 1, H, W]  (float32, 0‑1)
    labels : Tensor [N]           (int64)       – optional, may be None
    """
    bundle = torch.load(path, map_location="cpu")

    #  Adjust to the exact structure 
    if isinstance(bundle, dict):
        # e.g. {'data': Tensor, 'labels': Tensor}
        images = bundle["data"] if "data" in bundle else bundle["images"]
        labels = bundle.get("labels")
    elif isinstance(bundle, (list, tuple)):
        images, labels = bundle
    else:  # plain tensor
        images, labels = bundle, None

    # force shape [N,1,H,W] and float32
    if images.ndim == 3:              # [N,H,W]  → add channel
        images = images.unsqueeze(1)
    images = images.float() / images.max()  # ensure 0‑1
    return images, labels


