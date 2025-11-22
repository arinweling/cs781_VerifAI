"""
PyTorch-based classifier for VerifAI falsifier
Drop-in replacement for classifier.py using ONNX/PyTorch models

Usage:
    # Terminal 1: Start classifier server
    python classifier_pytorch.py
    
    # Terminal 2: Run falsifier
    python falsifier.py
"""

import numpy as np
from dotmap import DotMap
from verifai.client import Client
from renderer.kittiLib import getLib
from renderer.generator import genImage

# Configuration: Choose your inference backend
USE_ONNX_RUNTIME = True  # Recommended: Most reliable
USE_PYTORCH_NATIVE = False  # Alternative: If you have .pth file

if USE_ONNX_RUNTIME:
    try:
        from pytorch_model import ONNXInferenceWrapper
        print("✓ Using ONNX Runtime backend")
    except ImportError as e:
        print(f"Error importing ONNX Runtime: {e}")
        print("Please run: pip install onnxruntime")
        import sys
        sys.exit(1)
elif USE_PYTORCH_NATIVE:
    try:
        import torch
        from pytorch_model import CarDetectorCNN
        print("✓ Using PyTorch native backend")
    except ImportError as e:
        print(f"Error importing PyTorch: {e}")
        print("Please run: pip install torch torchvision")
        import sys
        sys.exit(1)
else:
    print("Error: No backend selected")
    print("Set either USE_ONNX_RUNTIME or USE_PYTORCH_NATIVE to True")
    import sys
    sys.exit(1)


class PyTorchClassifier(Client):
    """
    Classifier using PyTorch/ONNX models
    Compatible with VerifAI falsifier workflow
    """
    
    def __init__(self, classifier_data):
        port = classifier_data.port
        bufsize = classifier_data.bufsize
        super().__init__(port, bufsize)
        
        # Load the appropriate model
        if USE_ONNX_RUNTIME:
            print(f"Loading ONNX model from: {classifier_data.onnx_path}")
            self.model = ONNXInferenceWrapper(classifier_data.onnx_path)
            print("✓ ONNX model loaded successfully")
            
        elif USE_PYTORCH_NATIVE:
            print(f"Loading PyTorch model from: {classifier_data.pytorch_path}")
            self.model = CarDetectorCNN(num_classes=2, img_size=128)
            self.model.load_state_dict(torch.load(classifier_data.pytorch_path))
            self.model.eval()
            print("✓ PyTorch model loaded successfully")
        
        # Load KITTI library for image generation
        self.lib = getLib()
        print("✓ KITTI library loaded")
    
    def simulate(self, sample):
        """
        Simulate a sample by generating an image and classifying it
        
        Args:
            sample: Feature sample from VerifAI sampler
        
        Returns:
            Dictionary with 'yTrue' and 'yPred' keys
        """
        # Generate image from sample
        img, _ = genImage(self.lib, sample)
        
        # Ground truth: actual number of cars in the sample
        yTrue = len(sample.cars)
        
        # Prediction: classify the generated image
        predictions = self.model.predict(np.array(img))
        yPred = np.argmax(predictions[0]) + 1  # +1 because classes are 1-indexed
        
        # Return results
        res = {
            'yTrue': yTrue,
            'yPred': yPred
        }
        
        return res


# Configuration
PORT = 8888
BUFSIZE = 4096

classifier_data = DotMap()
classifier_data.port = PORT
classifier_data.bufsize = BUFSIZE

# Model paths
if USE_ONNX_RUNTIME:
    classifier_data.onnx_path = './data/car_detector/car-detector-model.onnx'
elif USE_PYTORCH_NATIVE:
    classifier_data.pytorch_path = './data/car_detector/car-detector-model.pth'

# You can also use the adversarial model if available:
# classifier_data.onnx_path = './data/adversarial_model/car-detector-model-adversarial.onnx'
# classifier_data.pytorch_path = './data/adversarial_model/car-detector-model-adversarial.pth'

# Create and run classifier
print("="*60)
print("PyTorch/ONNX Classifier Server")
print("="*60)
print(f"Port: {PORT}")
print(f"Buffer size: {BUFSIZE}")
print("="*60)
print("\nWaiting for falsifier connection...")
print("(Run 'python falsifier.py' in another terminal)")
print("="*60)

client_task = PyTorchClassifier(classifier_data)

# Main server loop
request_count = 0
while True:
    if not client_task.run_client():
        print("\n" + "="*60)
        print("End of all classifier calls")
        print(f"Total requests processed: {request_count}")
        print("="*60)
        break
    
    request_count += 1
    if request_count % 10 == 0:
        print(f"Processed {request_count} requests...")
