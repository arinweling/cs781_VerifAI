import sys, getopt
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from model import Model
import matplotlib.image as mpimg

# Change these to test different models:
# Original model:
# GRAPH_PATH = './data/car_detector/checkpoint/car-detector-model.meta'
# CHECKPOINT_PATH = './data/car_detector/checkpoint/'

# Adversarially retrained model:
GRAPH_PATH = './data/checkpoint/car-detector-model-adversarial.meta'
CHECKPOINT_PATH = './data/checkpoint/'

IMAGE_PATH = './counterexample_images/random_0.png'


with tf.Session() as sess:
    nn = Model()
    nn.init(GRAPH_PATH, CHECKPOINT_PATH, sess)
    image = cv2.imread(IMAGE_PATH)
    print(nn.predict(image)[0])
