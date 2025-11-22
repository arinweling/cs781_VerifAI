"""
Inspect TensorFlow model to find tensor names
This helps identify the correct input/output tensor names for conversion
"""

import tensorflow as tf

# Disable TF2 behavior
tf.compat.v1.disable_v2_behavior()

def inspect_model(checkpoint_path, meta_graph_path):
    """
    Inspect a TensorFlow model to find all tensor names
    
    Args:
        checkpoint_path: Path to checkpoint directory
        meta_graph_path: Path to .meta file
    """
    print("="*60)
    print("TensorFlow Model Inspector")
    print("="*60)
    
    tf.compat.v1.reset_default_graph()
    
    with tf.compat.v1.Session() as sess:
        # Load model
        print(f"\nLoading model from: {meta_graph_path}")
        saver = tf.compat.v1.train.import_meta_graph(meta_graph_path)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        
        graph = tf.compat.v1.get_default_graph()
        
        # Get all operations
        all_ops = graph.get_operations()
        
        print(f"\nTotal operations in graph: {len(all_ops)}")
        print("\n" + "="*60)
        print("PLACEHOLDERS (likely inputs):")
        print("="*60)
        
        placeholders = []
        for op in all_ops:
            if op.type == 'Placeholder':
                for output in op.outputs:
                    shape = output.get_shape()
                    dtype = output.dtype
                    placeholders.append({
                        'name': output.name,
                        'shape': shape,
                        'dtype': dtype,
                        'op_name': op.name
                    })
                    print(f"\nName: {output.name}")
                    print(f"  Shape: {shape}")
                    print(f"  Type: {dtype}")
        
        print("\n" + "="*60)
        print("OUTPUT OPERATIONS (likely outputs):")
        print("="*60)
        
        # Look for common output operation names
        output_keywords = ['pred', 'output', 'softmax', 'logits', 'result']
        potential_outputs = []
        
        for op in all_ops:
            op_name_lower = op.name.lower()
            # Check if operation name contains output keywords
            if any(keyword in op_name_lower for keyword in output_keywords):
                for output in op.outputs:
                    shape = output.get_shape()
                    dtype = output.dtype
                    potential_outputs.append({
                        'name': output.name,
                        'shape': shape,
                        'dtype': dtype,
                        'op_type': op.type
                    })
                    print(f"\nName: {output.name}")
                    print(f"  Shape: {shape}")
                    print(f"  Type: {dtype}")
                    print(f"  Op Type: {op.type}")
        
        print("\n" + "="*60)
        print("ALL TENSOR NAMES (first 50):")
        print("="*60)
        
        # Show all tensors (limited to first 50)
        for i, op in enumerate(all_ops[:50]):
            for output in op.outputs:
                print(f"{output.name:50} | Shape: {str(output.get_shape()):20} | Type: {op.type}")
        
        if len(all_ops) > 50:
            print(f"\n... and {len(all_ops) - 50} more operations")
        
        print("\n" + "="*60)
        print("RECOMMENDED TENSOR NAMES FOR CONVERSION:")
        print("="*60)
        
        # Find input tensor (typically a placeholder with 4D shape for images)
        input_tensor = None
        for p in placeholders:
            if len(p['shape']) == 4:  # Image input: [batch, height, width, channels]
                input_tensor = p['name']
                print(f"\nInput tensor:  {input_tensor}")
                print(f"  Shape: {p['shape']}")
                break
        
        # Find output tensor (look for softmax or pred in name)
        output_tensor = None
        for out in potential_outputs:
            if 'softmax' in out['name'].lower() or 'pred' in out['name'].lower():
                if 'cls' not in out['name'].lower():  # Skip class predictions
                    output_tensor = out['name']
                    print(f"\nOutput tensor: {output_tensor}")
                    print(f"  Shape: {out['shape']}")
                    break
        
        # If we found both, provide conversion command
        if input_tensor and output_tensor:
            print("\n" + "="*60)
            print("USE THESE IN YOUR CONVERSION SCRIPT:")
            print("="*60)
            print(f"\ninput_names=['{input_tensor}']")
            print(f"output_names=['{output_tensor}']")
        
        print("\n" + "="*60)
        
        return {
            'placeholders': placeholders,
            'potential_outputs': potential_outputs,
            'input_tensor': input_tensor,
            'output_tensor': output_tensor
        }


if __name__ == "__main__":
    checkpoint_path = './data/car_detector/checkpoint'
    meta_graph_path = './data/car_detector/checkpoint/car-detector-model.meta'
    
    result = inspect_model(checkpoint_path, meta_graph_path)
