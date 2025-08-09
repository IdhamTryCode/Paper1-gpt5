#!/usr/bin/env python3
"""
Comprehensive CNN Model Implementations

This module implements all specified CNN architectures with advanced features
for benchmarking between Python and Rust implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, Any


class ResNet18(nn.Module):
    """ResNet-18 architecture with comprehensive features."""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = False, 
                 dropout_rate: float = 0.5, use_batch_norm: bool = True):
        super(ResNet18, self).__init__()
        
        if pretrained:
            self.model = models.resnet18(pretrained=True)
            # Modify final layer for different number of classes
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            self.model = models.resnet18(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Add dropout for regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        
        # Add batch normalization if requested
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_classes)
        else:
            self.batch_norm = None
    
    def forward(self, x):
        x = self.model(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        return x


class VGG16(nn.Module):
    """VGG-16 architecture with comprehensive features."""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = False,
                 dropout_rate: float = 0.5, use_batch_norm: bool = True):
        super(VGG16, self).__init__()
        
        if pretrained:
            self.model = models.vgg16(pretrained=True)
            # Modify final layer for different number of classes
            self.model.classifier[-1] = nn.Linear(4096, num_classes)
        else:
            self.model = models.vgg16(pretrained=False)
            self.model.classifier[-1] = nn.Linear(4096, num_classes)
        
        # Add dropout for regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        
        # Add batch normalization if requested
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_classes)
        else:
            self.batch_norm = None
    
    def forward(self, x):
        x = self.model(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        return x


class MobileNet(nn.Module):
    """MobileNet architecture with comprehensive features."""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = False,
                 dropout_rate: float = 0.2, use_batch_norm: bool = True):
        super(MobileNet, self).__init__()
        
        if pretrained:
            self.model = models.mobilenet_v2(pretrained=True)
            # Modify final layer for different number of classes
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        else:
            self.model = models.mobilenet_v2(pretrained=False)
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        
        # Add dropout for regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        
        # Add batch normalization if requested
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_classes)
        else:
            self.batch_norm = None
    
    def forward(self, x):
        x = self.model(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        return x


class EnhancedLeNet(nn.Module):
    """Enhanced LeNet architecture with batch normalization and dropout."""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3,
                 use_batch_norm: bool = True):
        super(EnhancedLeNet, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        
        # Batch normalization layers
        if use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(6)
            self.batch_norm2 = nn.BatchNorm2d(16)
            self.batch_norm3 = nn.BatchNorm2d(32)
        else:
            self.batch_norm1 = None
            self.batch_norm2 = None
            self.batch_norm3 = None
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        if self.batch_norm1 is not None:
            x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second conv block
        x = self.conv2(x)
        if self.batch_norm2 is not None:
            x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Third conv block
        x = self.conv3(x)
        if self.batch_norm3 is not None:
            x = self.batch_norm3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)


class EnhancedSimpleCNN(nn.Module):
    """Enhanced Simple CNN with advanced layers and optimization."""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5,
                 use_batch_norm: bool = True, use_residual: bool = False):
        super(EnhancedSimpleCNN, self).__init__()
        
        self.use_residual = use_residual
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, padding=1)
        
        # Batch normalization layers
        if use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(32)
            self.batch_norm2 = nn.BatchNorm2d(64)
            self.batch_norm3 = nn.BatchNorm2d(128)
            self.batch_norm4 = nn.BatchNorm2d(256)
        else:
            self.batch_norm1 = None
            self.batch_norm2 = None
            self.batch_norm3 = None
            self.batch_norm4 = None
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.dropout3 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.dropout4 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        # First conv block
        identity = x
        x = self.conv1(x)
        if self.batch_norm1 is not None:
            x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.dropout1 is not None:
            x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        if self.batch_norm2 is not None:
            x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.dropout2 is not None:
            x = self.dropout2(x)
        
        # Third conv block
        x = self.conv3(x)
        if self.batch_norm3 is not None:
            x = self.batch_norm3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.dropout3 is not None:
            x = self.dropout3(x)
        
        # Fourth conv block
        x = self.conv4(x)
        if self.batch_norm4 is not None:
            x = self.batch_norm4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.dropout4 is not None:
            x = self.dropout4(x)
        
        # Global average pooling
        x = self.global_pool(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)


class AttentionCNN(nn.Module):
    """CNN with attention mechanism for enhanced performance."""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(AttentionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Batch normalization
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Apply attention mechanism
        batch_size, channels, height, width = x.size()
        x_flat = x.view(batch_size, channels, -1).transpose(1, 2)  # (batch, seq_len, channels)
        
        # Self-attention
        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        x = attn_output.transpose(1, 2).view(batch_size, channels, height, width)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)


def create_cnn_model(architecture: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """Factory function to create CNN models."""
    
    model_configs = {
        "resnet18": {
            "class": ResNet18,
            "default_params": {"pretrained": False, "dropout_rate": 0.5, "use_batch_norm": True}
        },
        "vgg16": {
            "class": VGG16,
            "default_params": {"pretrained": False, "dropout_rate": 0.5, "use_batch_norm": True}
        },
        "mobilenet": {
            "class": MobileNet,
            "default_params": {"pretrained": False, "dropout_rate": 0.2, "use_batch_norm": True}
        },
        "lenet": {
            "class": EnhancedLeNet,
            "default_params": {"dropout_rate": 0.3, "use_batch_norm": True}
        },
        "simple_cnn": {
            "class": EnhancedSimpleCNN,
            "default_params": {"dropout_rate": 0.5, "use_batch_norm": True, "use_residual": False}
        },
        "attention_cnn": {
            "class": AttentionCNN,
            "default_params": {"dropout_rate": 0.5}
        }
    }
    
    if architecture not in model_configs:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    config = model_configs[architecture]
    model_class = config["class"]
    default_params = config["default_params"].copy()
    
    # Update with provided parameters
    default_params.update(kwargs)
    default_params["num_classes"] = num_classes
    
    return model_class(**default_params)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get comprehensive information about a model."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": size_mb,
        "architecture": model.__class__.__name__
    } 