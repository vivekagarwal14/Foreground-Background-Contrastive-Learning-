import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization
        # Reduce dimensionality with Global Average Pooling instead of flattening everything
        self.global_avg_pool = nn.AdaptiveAvgPool2d((8, 8))  # Reduces 64x64 to 8x8
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)  # Reduced FC layer size
        self.dropout = nn.Dropout(0.3)  # Dropout to prevent overfitting
        self.fc2 = nn.Linear(1024, 256 * 256)

    def forward(self, x):
        # Convolution + BatchNorm + Activation + MaxPooling
        x = F.gelu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.gelu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        # Global Average Pooling to reduce feature size
        x = self.global_avg_pool(x)
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        # Fully Connected + Dropout
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        # Output layer with sigmoid (or tanh)
        x = torch.sigmoid(self.fc2(x))
        # Reshape output to 256x256
        x = x.view(-1, 1, 256, 256)
        return x


def get_model(model_name="simple_cnn", **kwargs):
    """
    Factory function to get the specified model.
    
    Args:
        model_name (str): Name of the model to create
        **kwargs: Additional arguments for model initialization
        
    Returns:
        torch.nn.Module: The requested model
    """
    models = {
        'simple_cnn': GlomeruliDenoiserCNN,
        'net': GlomeruliDenoiserCNN,  # Keep backward compatibility
        # Add more models here as you create them:
        # 'unet': UNetDenoiser,
        # 'resnet': ResNetDenoiser,
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not available. Choose from {list(models.keys())}")
    
    return models[model_name](**kwargs)