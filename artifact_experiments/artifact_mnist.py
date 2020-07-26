"""artifact_mnist.py
~~~~~~~~~~~~~~~~~~~~
Charles Sedgwick

Take the 50,000 MNIST training images, and create an expanded set images by introducing a given percentage of random
artifacts to a certain percentage of the images.
Save the resulting file to ../data/mnist_artifact_$PERCENTIMAGES_$PERCENTARTIFACT.pkl.gz.

Note that this program is memory intensive, and may not run on small
systems.

Based off of expand_mnist.py by Micheal Nielsen

"""

from __future__ import print_function

#### Libraries

# Standard library
import pickle
import gzip
import os.path
import random
import math

# Third-party libraries
import numpy as np

def generate_set(ratio_images, ratio_artifacts):

    print("Creating artifact MNIST set 1: "+str(ratio_images*100)+" of images with "+str(ratio_artifacts*100)+" of image made up of artifacts")

    percent_imgs = math.ceil(ratio_images * 100)
    percent_art = math.ceil(ratio_artifacts * 100)

    if os.path.exists("../data/mnist_artifact"+str(int(percent_imgs))+"_"+str(int(percent_art))+".pkl.gz"):
        print("The artifact training set already exists.  Exiting.")
    else:
        f = gzip.open("../data/mnist.pkl.gz", 'rb')
        training_data, validation_data, test_data = pickle.load(f)
        f.close()

        # Process training data
        artifact_training_pairs = []
        # determine number of images to alter
        num_images = math.ceil(50000*ratio_images)
        # alter the first num_images images in array
        for j in range(0, int(num_images)):
            k = 0
            # randomly sample pixels
            artifact_pixels = random.sample(xrange(784), int(784*ratio_artifacts))
            # turn sampled pixels into artifacts
            for i in artifact_pixels:
                training_data[0][j, i] = random.random()

        for x, y in zip(training_data[0], training_data[1]):
            artifact_training_pairs.append((x, y))
        # shuffle training data
        random.shuffle(artifact_training_pairs)
        artifact_training_data = [list(d) for d in zip(*artifact_training_pairs)]

        # Process validation data
        artifact_validation_pairs = []
        # determine number of images to alter
        num_images = math.ceil(10000 * ratio_images)
        # alter the first num_images images in array
        for j in range(0, int(num_images)):
            k = 0
            # randomly sample pixels
            artifact_pixels = random.sample(xrange(784), int(784 * ratio_artifacts))
            # turn sampled pixels into artifacts
            for i in artifact_pixels:
                validation_data[0][j, i] = random.random()

        for x, y in zip(validation_data[0], validation_data[1]):
            artifact_validation_pairs.append((x, y))
        # shuffle validation data
        random.shuffle(artifact_validation_pairs)
        artifact_validation_data = [list(d) for d in zip(*artifact_validation_pairs)]

        # Process test data
        artifact_test_pairs = []
        # determine number of images to alter
        num_images = math.ceil(10000*ratio_images)
        # alter the first num_images images in array
        for j in range(0, int(num_images)):
            k = 0
            # randomly sample pixels
            artifact_pixels = random.sample(xrange(784), int(784*ratio_artifacts))
            # turn sampled pixels into artifacts
            for i in artifact_pixels:
                test_data[0][j, i] = random.random()

        for x, y in zip(test_data[0], test_data[1]):
            artifact_test_pairs.append((x, y))
        # shuffle training data
        random.shuffle(artifact_test_pairs)
        artifact_test_data = [list(d) for d in zip(*artifact_test_pairs)]
        print("Saving artifact data. This may take a few minutes.")
        f = gzip.open("../data/mnist_artifact_"+str(int(percent_imgs))+"_"+str(int(percent_art))+".pkl.gz", "w")
        pickle.dump((artifact_training_data, validation_data, artifact_test_data), f)
        f.close()


# generate new versions of MNIST data
generate_set(0.1, 0.1)
generate_set(0.1, 0.25)
generate_set(0.25, 0.1)
generate_set(0.25, 0.25)
generate_set(0.5, 0.1)
generate_set(0.5, 0.25)
generate_set(0.5, 0.5)
