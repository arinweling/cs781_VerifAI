"""
Convert TensorFlow car detector model to PyTorch via ONNX

This script performs the following steps:
1. Loads the TensorFlow model from checkpoint
2. Exports it to ONNX format
3. Loads the ONNX model in PyTorch
4. Saves the PyTorch model

Requirements:
    pip install tensorflow onnx torch torchvision tf2onnx onnx2pytorch
"""

import tensorflow as tf
import tf2onnx
import onnx
import torch
import numpy as np
from PIL import Image
import os

# Disable TF2 behavior for compatibility
tf.compat.v1.disable_v2_behavior()


def export_tf_to_onnx(checkpoint_path, meta_graph_path, output_onnx_path, 
                      input_tensor_name="x:0", output_tensor_name="yPred:0"):
    """
    Export TensorFlow model to ONNX format
    
    Args:
        checkpoint_path: Path to TensorFlow checkpoint directory
        meta_graph_path: Path to .meta file
        output_onnx_path: Output path for ONNX model
        input_tensor_name: Name of input tensor (default: "x:0" for car detector)
        output_tensor_name: Name of output tensor (default: "yPred:0" for car detector)
    """
    print("Step 1: Loading TensorFlow model...")
    print(f"  Input tensor: {input_tensor_name}")
    print(f"  Output tensor: {output_tensor_name}")
    
    # Reset default graph
    tf.compat.v1.reset_default_graph()
    
    # Create a new session
    with tf.compat.v1.Session() as sess:
        # Load the meta graph
        saver = tf.compat.v1.train.import_meta_graph(meta_graph_path)
        
        # Restore weights
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        
        # Get the default graph
        graph = tf.compat.v1.get_default_graph()
        
        # Get input and output tensors
        try:
            x = graph.get_tensor_by_name(input_tensor_name)
            y_pred = graph.get_tensor_by_name(output_tensor_name)
        except KeyError as e:
            print(f"\n❌ Error: Tensor not found: {e}")
            print("\nAvailable tensors in the graph:")
            print("="*60)
            for op in graph.get_operations()[:20]:  # Show first 20
                for output in op.outputs:
                    print(f"  {output.name:40} | Shape: {output.get_shape()}")
            print("  ... (use inspect_tf_model.py to see all tensors)")
            print("="*60)
            raise
        
        print(f"Input tensor shape: {x.shape}")
        print(f"Output tensor shape: {y_pred.shape}")
        
        print("\nStep 2: Converting to ONNX...")
        
        # Convert to ONNX using the updated tf2onnx API
        try:
            # Try newer API first (tf2onnx >= 1.9)
            import tf2onnx.tfonnx
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(
                sess.graph,
                input_names=[input_tensor_name],
                output_names=[output_tensor_name],
                opset=13
            )
            onnx_model = onnx_graph[0]
        except (AttributeError, ImportError):
            # Fallback to older API
            from tf2onnx import tf_loader
            graph_def = sess.graph.as_graph_def()
            
            with tf.Graph().as_default() as tf_graph:
                tf.import_graph_def(graph_def, name='')
            
            with tf.compat.v1.Session(graph=tf_graph) as new_sess:
                onnx_model = tf2onnx.convert.from_graph_def(
                    graph_def,
                    input_names=[input_tensor_name],
                    output_names=[output_tensor_name],
                    opset=13
                )[0]
        
        # Save ONNX model
        with open(output_onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"✓ ONNX model saved to: {output_onnx_path}")
    
    return output_onnx_path


def create_pytorch_model_from_onnx(onnx_path):
    """
    Create PyTorch model from ONNX
    
    Args:
        onnx_path: Path to ONNX model
    
    Returns:
        PyTorch model
    """
    print("\nStep 3: Loading ONNX model into PyTorch...")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Option 1: Use onnx2pytorch (automatic conversion)
    try:
        from onnx2pytorch import ConvertModel
        pytorch_model = ConvertModel(onnx_model)
        print("✓ Successfully converted to PyTorch using onnx2pytorch")
        return pytorch_model, "onnx2pytorch"
    except ImportError:
        print("⚠ onnx2pytorch not available, will use onnxruntime instead")
        return None, "onnxruntime"


def save_pytorch_model(model, output_path):
    """
    Save PyTorch model
    
    Args:
        model: PyTorch model
        output_path: Output path for PyTorch model
    """
    print(f"\nStep 4: Saving PyTorch model to {output_path}...")
    torch.save(model.state_dict(), output_path)
    print("✓ PyTorch model saved")


def verify_conversion(tf_checkpoint_path, tf_meta_path, onnx_path, pytorch_model=None):
    """
    Verify that the conversion is correct by comparing outputs
    
    Args:
        tf_checkpoint_path: Path to TensorFlow checkpoint
        tf_meta_path: Path to .meta file
        onnx_path: Path to ONNX model
        pytorch_model: PyTorch model (optional)
    """
    print("\nStep 5: Verifying conversion...")
    
    # Create a dummy input image
    img_size = 128
    num_channels = 3
    dummy_input = np.random.rand(1, img_size, img_size, num_channels).astype(np.float32)
    
    # Get TensorFlow output
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(tf_meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(tf_checkpoint_path))
        graph = tf.compat.v1.get_default_graph()
        
        x = graph.get_tensor_by_name("x:0")
        y_pred = graph.get_tensor_by_name("yPred:0")
        y_true = graph.get_tensor_by_name("yTrue:0")
        
        y_test = np.zeros((1, 2))
        feed_dict = {x: dummy_input, y_true: y_test}
        tf_output = sess.run(y_pred, feed_dict=feed_dict)
    
    print(f"TensorFlow output: {tf_output}")
    
    # Get ONNX output using onnxruntime
    import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {'x:0': dummy_input})[0]
    print(f"ONNX output: {onnx_output}")
    
    # Compare outputs
    diff = np.abs(tf_output - onnx_output).max()
    print(f"\nMaximum difference between TF and ONNX: {diff}")
    
    if diff < 1e-5:
        print("✓ Conversion verified successfully! Models produce identical outputs.")
    else:
        print("⚠ Warning: Outputs differ. Please check the conversion.")
    
    # If PyTorch model is available, verify it too
    if pytorch_model is not None:
        pytorch_model.eval()
        with torch.no_grad():
            # Note: PyTorch expects NCHW format, TensorFlow uses NHWC
            pytorch_input = torch.from_numpy(dummy_input.transpose(0, 3, 1, 2))
            pytorch_output = pytorch_model(pytorch_input).numpy()
            
            print(f"PyTorch output: {pytorch_output}")
            diff_pytorch = np.abs(tf_output - pytorch_output).max()
            print(f"Maximum difference between TF and PyTorch: {diff_pytorch}")
            
            if diff_pytorch < 1e-5:
                print("✓ PyTorch conversion verified!")
            else:
                print("⚠ Warning: PyTorch outputs differ.")


def main():
    # Paths
    checkpoint_path = './data/car_detector/checkpoint'
    meta_graph_path = './data/car_detector/checkpoint/car-detector-model.meta'
    output_onnx_path = './data/car_detector/car-detector-model.onnx'
    output_pytorch_path = './data/car_detector/car-detector-model.pth'
    
    # Tensor names specific to this car detector model
    # These are defined in model/model.py with name='x' and name='yPred'
    input_tensor_name = "x:0"
    output_tensor_name = "yPred:0"
    
    print("="*60)
    print("TensorFlow to PyTorch Conversion (via ONNX)")
    print("="*60)
    print(f"\nModel-specific tensor names:")
    print(f"  Input:  {input_tensor_name}")
    print(f"  Output: {output_tensor_name}")
    print(f"\nNote: These names are defined in model/model.py")
    print(f"For different models, use inspect_tf_model.py to find tensor names")
    print("="*60)
    
    # Step 1 & 2: Export TF to ONNX
    onnx_path = export_tf_to_onnx(
        checkpoint_path, 
        meta_graph_path, 
        output_onnx_path,
        input_tensor_name,
        output_tensor_name
    )
    
    # Step 3: Load ONNX into PyTorch
    pytorch_model, method = create_pytorch_model_from_onnx(onnx_path)
    
    # Step 4: Save PyTorch model (if conversion successful)
    if pytorch_model is not None:
        save_pytorch_model(pytorch_model, output_pytorch_path)
    
    # Step 5: Verify conversion
    verify_conversion(checkpoint_path, meta_graph_path, onnx_path, pytorch_model)
    
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - ONNX model: {output_onnx_path}")
    if pytorch_model is not None:
        print(f"  - PyTorch model: {output_pytorch_path}")
    else:
        print(f"\nNote: PyTorch model not created. Using ONNX with onnxruntime.")
        print(f"You can still use the ONNX model with PyTorch using onnxruntime or onnx-pytorch.")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. To use ONNX model with onnxruntime:")
    print("   import onnxruntime as ort")
    print("   session = ort.InferenceSession('car-detector-model.onnx')")
    print("   output = session.run(None, {'x:0': input_data})")
    print("\n2. To use PyTorch model:")
    print("   import torch")
    print("   model = torch.load('car-detector-model.pth')")
    print("   output = model(input_tensor)")
    print("="*60)


if __name__ == "__main__":
    main()
