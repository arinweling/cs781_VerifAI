"""
Prepare adversarial examples (incorrectly classified images) for retraining.

This script:
1. Reads all error samples from the error table
2. Generates images with their TRUE labels (not predicted labels)
3. Saves them in the correct folder structure for training
4. Optionally copies existing training data to create an augmented dataset
"""

from verifai.features.features import *
from verifai.samplers.feature_sampler import *
from verifai.falsifier import generic_falsifier
from verifai.monitor import specification_monitor
from dotmap import DotMap
from renderer.generator import genImage
from renderer.kittiLib import getLib
import pickle
import os
import shutil
from PIL import Image

# # Load the saved pickle file from a previous falsifier run
# print("Loading samples from pickle file...")
# with open("generated_samples.pickle", "rb") as f:
#     all_samples = pickle.load(f)

# Recreate the space (same as in falsifier.py)
carDomain = Struct({
    'xPos': Box([0, 1]),
    'yPos': Box([0, 1]),
    'carID': Categorical(*np.arange(0,37))
})

space = FeatureSpace({
    'backgroundID': Feature(Categorical(*np.arange(0, 35))),
    'cars': Feature(Array(carDomain, (2,))),
    'brightness': Feature(Box([0.5, 1])),
    'sharpness': Feature(Box([0, 1])),
    'contrast': Feature(Box([0.5, 1.5])),
    'color': Feature(Box([0, 1]))
})

# Option 1: Load error table from a saved falsifier run
# You'll need to save the falsifier object or error table separately
# For now, we'll create a simple approach

def prepare_adversarial_data(output_dir='data/adversarial_train/', 
                             error_images_dir='counterexample_images/all_images/',
                             merge_with_original=False):
    """
    Prepare adversarial training data from error samples.
    
    Args:
        output_dir: Where to save the organized training data
        error_images_dir: Where the error images are stored
        merge_with_original: If True, copy original training data too
    """
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, '1'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '2'), exist_ok=True)
    
    # If merging, copy original training data first
    if merge_with_original and os.path.exists('data/train/'):
        print("Copying original training data...")
        for class_folder in ['1', '2']:
            src_dir = os.path.join('data/train/', class_folder)
            dst_dir = os.path.join(output_dir, class_folder)
            if os.path.exists(src_dir):
                for img_file in os.listdir(src_dir):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        shutil.copy(
                            os.path.join(src_dir, img_file),
                            os.path.join(dst_dir, f"orig_{img_file}")
                        )
        print(f"Original data copied to {output_dir}")
    
    print(f"\nNow processing adversarial examples...")
    print(f"This approach requires you to have run falsifier and saved error images")
    print(f"Since we need the TRUE labels, we'll regenerate from the error table\n")


def prepare_from_falsifier_object(falsifier, output_dir='data/adversarial_train/', 
                                  merge_with_original=False):
    """
    Better approach: Use the falsifier object directly to get true labels.
    
    Call this from your falsifier.py script after running falsification.
    """
    
    # Clear the output directory before starting
    if os.path.exists(output_dir):
        print(f"Cleaning up existing adversarial training data in {output_dir}...")
        shutil.rmtree(output_dir)
    
    # Create fresh output directories
    os.makedirs(os.path.join(output_dir, '1'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '2'), exist_ok=True)
    print(f"Created fresh directory: {output_dir}")
    
    # If merging, copy original training data first
    if merge_with_original and os.path.exists('data/train/'):
        print("Copying original training data...")
        for class_folder in ['1', '2']:
            src_dir = os.path.join('data/train/', class_folder)
            dst_dir = os.path.join(output_dir, class_folder)
            if os.path.exists(src_dir):
                for img_file in os.listdir(src_dir):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        shutil.copy(
                            os.path.join(src_dir, img_file),
                            os.path.join(dst_dir, f"orig_{img_file}")
                        )
        
        # Count original images
        count_1 = len([f for f in os.listdir(os.path.join(output_dir, '1')) if f.endswith(('.png', '.jpg'))])
        count_2 = len([f for f in os.listdir(os.path.join(output_dir, '2')) if f.endswith(('.png', '.jpg'))])
        print(f"Copied {count_1} images with 1 car, {count_2} images with 2 cars")
    
    # Get error samples
    print("\nProcessing adversarial examples from error table...")
    lib = getLib()
    
    all_error_indices = list(range(len(falsifier.error_table.table)))
    all_error_samples = falsifier.error_table.get_samples_by_index(all_error_indices)
    
    # Recreate space from falsifier
    space = falsifier.error_table.space
    
    count_adv = {1: 0, 2: 0}
    
    for idx, row in all_error_samples.iterrows():
        # Convert DataFrame row back to sample format
        sample = space.unflatten(row.drop('rho').values)
        
        # Get TRUE label (not predicted)
        true_label = len(sample.cars)  # Number of cars in the sample
        
        # Generate image
        img, _ = genImage(lib, sample)
        
        # Save to correct class folder based on TRUE label
        output_path = os.path.join(output_dir, str(true_label), f"adv_{idx}.png")
        img.save(output_path)
        count_adv[true_label] += 1
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(all_error_samples)} adversarial examples")
    
    print(f"\nAdversarial examples saved:")
    print(f"  Class 1 (1 car): {count_adv.get(1, 0)} images")
    print(f"  Class 2 (2 cars): {count_adv.get(2, 0)} images")
    
    # Print final counts
    final_count_1 = len([f for f in os.listdir(os.path.join(output_dir, '1')) if f.endswith('.png')])
    final_count_2 = len([f for f in os.listdir(os.path.join(output_dir, '2')) if f.endswith('.png')])
    
    print(f"\nTotal dataset in {output_dir}:")
    print(f"  Class 1: {final_count_1} images")
    print(f"  Class 2: {final_count_2} images")
    print(f"\nYou can now retrain using: python model/train.py")
    print(f"(Update trainPath in train.py to '{output_dir}')")


if __name__ == "__main__":
    print("This script should be called from falsifier.py")
    print("Add this at the end of your falsifier.py:")
    print()
    print("from prepare_adversarial_training_data import prepare_from_falsifier_object")
    print("prepare_from_falsifier_object(falsifier, merge_with_original=True)")
