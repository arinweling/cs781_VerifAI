"""
PyTorch implementation of the car detector model

This file contains:
1. A PyTorch model that mirrors the TensorFlow architecture
2. Functions to load weights from ONNX
3. Inference functions compatible with the existing classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


class CarDetectorCNN(nn.Module):
    """
    PyTorch implementation of the car detector CNN
    
    Architecture matches the TensorFlow model in model/model.py:
    - Conv1: 3x3, 32 filters
    - Conv2: 3x3, 32 filters  
    - Conv3: 3x3, 64 filters
    - FC1: 128 units
    - FC2: 2 units (num_classes)
    """
    
    def __init__(self, num_classes=2, img_size=128):
        super(CarDetectorCNN, self).__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        # Max pooling (applied after each conv layer in TF model)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened size after 3 pooling operations
        # Each pool reduces dimensions by half: 128 -> 64 -> 32 -> 16
        self.flattened_size = 16 * 16 * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
               Note: PyTorch uses NCHW format, unlike TensorFlow's NHWC
        
        Returns:
            Softmax probabilities of shape (batch, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.pool(x)
        x = F.relu(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.pool(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(-1, self.flattened_size)
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        # Softmax
        x = F.softmax(x, dim=1)
        
        return x
    
    def predict(self, image):
        """
        Predict for a single image (compatible with TF model interface)
        
        Args:
            image: PIL Image or numpy array (H, W, C) in RGB format
        
        Returns:
            Softmax probabilities as numpy array
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image, 'RGB')
        
        # Resize to model input size
        image = image.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype('float32') / 255.0
        
        # Convert from HWC to CHW format (PyTorch convention)
        image_array = image_array.transpose(2, 0, 1)
        
        # Add batch dimension and convert to tensor
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        
        # Run inference
        self.eval()
        with torch.no_grad():
            output = self.forward(image_tensor)
        
        return output.numpy()


def load_from_onnx(onnx_path, pytorch_model=None):
    """
    Load weights from ONNX model into PyTorch model
    
    Args:
        onnx_path: Path to ONNX model file
        pytorch_model: PyTorch model instance (if None, creates new one)
    
    Returns:
        PyTorch model with loaded weights
    """
    try:
        from onnx2pytorch import ConvertModel
        import onnx
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to PyTorch
        if pytorch_model is None:
            pytorch_model = ConvertModel(onnx_model)
        else:
            # Manual weight transfer (if needed)
            print("Warning: Manual weight transfer not implemented")
            print("Using automatic conversion instead")
            pytorch_model = ConvertModel(onnx_model)
        
        return pytorch_model
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install onnx2pytorch: pip install onnx2pytorch")
        return None


def load_from_state_dict(checkpoint_path, pytorch_model=None):
    """
    Load PyTorch model from saved state dict
    
    Args:
        checkpoint_path: Path to .pth file
        pytorch_model: PyTorch model instance (if None, creates new one)
    
    Returns:
        PyTorch model with loaded weights
    """
    if pytorch_model is None:
        pytorch_model = CarDetectorCNN()
    
    pytorch_model.load_state_dict(torch.load(checkpoint_path))
    pytorch_model.eval()
    
    return pytorch_model


class ONNXInferenceWrapper:
    """
    Wrapper for using ONNX model directly with onnxruntime
    This is useful if onnx2pytorch conversion fails
    """
    
    def __init__(self, onnx_path):
        """
        Initialize ONNX inference session
        
        Args:
            onnx_path: Path to ONNX model
        """
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(onnx_path)
            self.img_size = 128
            print(f"✓ ONNX model loaded from {onnx_path}")
        except ImportError:
            raise ImportError("Please install onnxruntime: pip install onnxruntime")
    
    def predict(self, image):
        """
        Predict for a single image
        
        Args:
            image: PIL Image or numpy array (H, W, C) in RGB format
        
        Returns:
            Softmax probabilities as numpy array
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image, 'RGB')
        
        # Resize to model input size
        image = image.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype('float32') / 255.0
        
        # Add batch dimension (ONNX expects NHWC format like TensorFlow)
        image_array = image_array.reshape(1, self.img_size, self.img_size, 3)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name: image_array})
        
        return result[0]


# Example usage
if __name__ == "__main__":
    print("PyTorch Car Detector Model")
    print("=" * 60)
    
    # Create model
    model = CarDetectorCNN(num_classes=2, img_size=128)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output (softmax probabilities): {output}")
    
    print("\nModel architecture:")
    print(model)
