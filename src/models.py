import torch
import torch.nn as nn
import torch.nn.functional as F


class GlomeruliDenoiserCNN(nn.Module):
    """
    Simple CNN denoiser for glomeruli images.
    Architecture: Conv2d -> BatchNorm -> GlobalAvgPool -> FC layers
    """
    def __init__(self):
        super(GlomeruliDenoiserCNN, self).__init__()
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


class GlomeruliDenoiserSimple(nn.Module):
    """
    Simple CNN denoiser without batch normalization and pooling.
    Architecture: Conv2d -> MaxPool -> FC layers
    """
    def __init__(self):
        super(GlomeruliDenoiserSimple, self).__init__()
        # First convolutional layer: 1 input channel, 32 output channels, kernel size 5, padding 2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        # Second convolutional layer: 32 input channels, 64 output channels, kernel size 5, padding 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # Calculate the size of the feature map after conv2 and two max-pooling operations
        # For a 256x256 input, after two 2x2 max-poolings, the feature map will be 64x64
        # So we flatten this to 64 * 64 * 64 = 262144 for the fully connected layer
        self.fc1 = nn.Linear(64 * 64 * 64, 1024)
        self.fc2 = nn.Linear(1024, 256 * 256)  # Adjusted output to match the 256x256 output size

    def forward(self, x):
        # Apply first convolution, followed by ReLU and 2x2 max-pooling
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2), negative_slope=0.001)
        x = F.gelu(F.max_pool2d(self.conv1(x), 2))
        
        # Apply second convolution, followed by ReLU and 2x2 max-pooling
        #x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.leaky_relu(F.max_pool2d(self.conv2(x), 2), negative_slope=0.001)
        x = F.gelu(F.max_pool2d(self.conv2(x), 2))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 64 * 64)  # Flatten to match the input to fc1
        # Apply the fully connected layers with ReLU activation
        #x = F.relu(self.fc1(x))
        #x = F.leaky_relu(self.fc1(x), negative_slope=0.001)
        x = F.gelu(self.fc1(x))
        
        # Apply sigmoid activation for the output
        x = torch.sigmoid(self.fc2(x))
        # Reshape the output to match the desired shape: batch_size x 1 x 256 x 256
        x = x.view(-1, 1, 256, 256)
        return x


class GlomeruliUNet(nn.Module):
    """
    U-Net architecture for glomeruli denoising.
    Features encoder-decoder structure with skip connections.
    """
    def __init__(self):
        super(GlomeruliUNet, self).__init__()
        # Encoder: Downsampling path
        self.enc1 = self.conv_block(1, 64, (256, 256))   # Input: 1x256x256 -> 64x256x256
        self.enc2 = self.conv_block(64, 128, (128, 128)) # 128x128 -> 128x128x128
        self.enc3 = self.conv_block(128, 256, (64, 64))  # 64x64 -> 256x64x64
        self.enc4 = self.conv_block(256, 512, (32, 32))  # 32x32 -> 512x32x32
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024, (16, 16)) # 16x16 -> 1024x16x16
        # Decoder: Upsampling path with skip connections
        self.upconv4 = self.upconv(1024, 512) # 32x32
        self.dec4 = self.conv_block(1024, 512, (32, 32)) # Concatenate with enc4
        self.upconv3 = self.upconv(512, 256) # 64x64
        self.dec3 = self.conv_block(512, 256, (64, 64)) # Concatenate with enc3
        self.upconv2 = self.upconv(256, 128) # 128x128
        self.dec2 = self.conv_block(256, 128, (128, 128)) # Concatenate with enc2
        self.upconv1 = self.upconv(128, 64) # 256x256
        self.dec1 = self.conv_block(128, 64, (256, 256)) # Concatenate with enc1
        # Final output layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  # Output: 1x256x256

    def conv_block(self, in_channels, out_channels, image_shape):
        """Convolutional block with two Conv layers + LayerNorm + ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.LayerNorm([out_channels, *image_shape]),
            #nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.LayerNorm([out_channels, *image_shape]),
            #nn.ReLU(inplace=True)
            nn.GELU()
        )

    def upconv(self, in_channels, out_channels):
        """Upsampling with transpose convolution"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)  # 64x256x256
        enc2 = self.enc2(F.max_pool2d(enc1, 2))  # 128x128x128
        enc3 = self.enc3(F.max_pool2d(enc2, 2))  # 256x64x64
        enc4 = self.enc4(F.max_pool2d(enc3, 2))  # 512x32x32
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))  # 1024x16x16
        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)  # 512x32x32
        dec4 = torch.cat((dec4, enc4), dim=1)  # Concatenate with enc4
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)  # 256x64x64
        dec3 = torch.cat((dec3, enc3), dim=1)  # Concatenate with enc3
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)  # 128x128x128
        dec2 = torch.cat((dec2, enc2), dim=1)  # Concatenate with enc2
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)  # 64x256x256
        dec1 = torch.cat((dec1, enc1), dim=1)  # Concatenate with enc1
        dec1 = self.dec1(dec1)
        # Final convolution
        output = self.final_conv(dec1)  # 1x256x256
        output = torch.log(1 + torch.exp(output))
        output = output.clamp_(min=0.)
        return output


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
        'simple_cnn': GlomeruliDenoiserCNN,      # With batch norm and global avg pooling
        'basic_cnn': GlomeruliDenoiserSimple,    # Simple version without batch norm
        'unet': GlomeruliUNet,                   # U-Net with skip connections
        'net': GlomeruliDenoiserSimple,          # Keep backward compatibility
        # Add more models here as you create them:
        # 'resnet': ResNetDenoiser,
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not available. Choose from {list(models.keys())}")
    
    return models[model_name](**kwargs)