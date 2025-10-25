"""
Retrain the car detector model with adversarial examples.

This script:
1. Prepares adversarial training data from error samples
2. Retrains the model on combined original + adversarial data
"""

from model import dataset, utils
from model.model import Model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# Configuration
CHECKPOINT_NAME = 'data/checkpoint/car-detector-model-adversarial'
TRAIN_PATH = 'data/adversarial_train/'  # Directory with combined data
CLASSES = ['1', '2']
IMG_SIZE = 128
NUM_CHANNELS = 3
VALIDATION_SIZE = 0.2
BATCH_SIZE = 2
NUM_ITERATIONS = 1000  # Reduced for fine-tuning (was 5000 for training from scratch)
                       # Fine-tuning needs fewer iterations since starting from good weights

def retrain():
    """Retrain the model with adversarial examples"""
    
    print("=" * 60)
    print("RETRAINING CAR DETECTOR WITH ADVERSARIAL EXAMPLES")
    print("=" * 60)
    
    # Load training and validation data
    print(f"\nLoading data from: {TRAIN_PATH}")
    data = dataset.readTrainSets(
        TRAIN_PATH, IMG_SIZE, CLASSES, validationSize=VALIDATION_SIZE)
    
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print(f"Training samples:   {len(data.train.labels)}")
    print(f"Validation samples: {len(data.valid.labels)}")
    print("=" * 60 + "\n")
    
    session = tf.Session()
    
    # Get model graph
    nn = Model()
    x, layerFc2, yTrue, yTrueCls, yPred, yPredCls = nn.getGraph(len(CLASSES))
    
    # Training setup
    crossEntropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=layerFc2, labels=yTrue)
    cost = tf.reduce_mean(crossEntropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    correct_prediction = tf.equal(yPredCls, yTrueCls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    session.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    
    # Load pretrained weights to fine-tune (recommended for adversarial retraining)
    # Comment out this block if you want to train from scratch instead
    try:
        saver.restore(session, tf.train.latest_checkpoint('data/car_detector/checkpoint/'))
        print("✓ Loaded pretrained weights. Fine-tuning on adversarial examples...\n")
    except Exception as e:
        print(f"⚠ Could not load pretrained weights: {e}")
        print("Training from scratch instead...\n")
    
    totalIterations = 0
    
    def showProgress(epoch, feedDictTrain, feedDictValidate, valLoss):
        acc = session.run(accuracy, feed_dict=feedDictTrain)
        val_acc = session.run(accuracy, feed_dict=feedDictValidate)
        msg = "Epoch {0:3d} | Train Acc: {1:6.1%} | Val Acc: {2:6.1%} | Val Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, valLoss))
    
    print("Starting training...")
    print("-" * 60)
    
    for i in range(NUM_ITERATIONS):
        # Fetch batch
        xBatch, yTrueBatch, _, _ = data.train.nextBatch(BATCH_SIZE)
        xValidBatch, yValidBatch, _, _ = data.valid.nextBatch(BATCH_SIZE)
        
        feedDictTr = {x: xBatch, yTrue: yTrueBatch}
        feedDictVal = {x: xValidBatch, yTrue: yValidBatch}
        
        session.run(optimizer, feed_dict=feedDictTr)
        
        # Show progress and save
        if i % int(data.train.num_examples / BATCH_SIZE) == 0:
            valLoss = session.run(cost, feed_dict=feedDictVal)
            epoch = int(i / int(data.train.num_examples / BATCH_SIZE))
            
            showProgress(epoch, feedDictTr, feedDictVal, valLoss)
            saver.save(session, CHECKPOINT_NAME)
    
    print("-" * 60)
    print(f"\nTraining complete! Model saved to: {CHECKPOINT_NAME}")
    print("\nTo use the retrained model, update classifier.py:")
    print(f"  checkpoint_path = '{CHECKPOINT_NAME.rsplit('/', 1)[0]}/'")
    

if __name__ == "__main__":
    import os
    if not os.path.exists(TRAIN_PATH):
        print(f"ERROR: Training data not found at {TRAIN_PATH}")
        print("\nFirst run:")
        print("  1. Run falsifier.py to generate error samples")
        print("  2. Add this to the end of falsifier.py:")
        print("       from prepare_adversarial_training_data import prepare_from_falsifier_object")
        print("       prepare_from_falsifier_object(falsifier, merge_with_original=True)")
        print("  3. Then run this script")
    else:
        retrain()
