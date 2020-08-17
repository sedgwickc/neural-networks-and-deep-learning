# -----------------------------------------------------------------------------
# artifact_experiments.py
# This script contains the experiments used to test the ability of a basic ANN
# and deep learning ANN described by Nielsen. It also generates graphs of the
# performance of the best out of 3 networks trained of each type.


import src.network3 as network3
import src.conv as conv

data_artifact = "../data/mnist_artifact_25_25.pkl.gz"
data_orig = "../data/mnist.pkl.gz"
mini_batch_size = 10

print("Running shallow network experiments...")

print("Training with original data...")
# Nielson achieved 97.8 percent accuracy
nets_shallow = conv.shallow(epochs=60, data=data_orig)
error_locations_shallow, erroneous_predictions_shallow = conv.ensemble(nets_shallow)
plt = conv.plot_errors(error_locations_shallow, erroneous_predictions_shallow)
plt.savefig("results/shallow_errors_orig.png")

print("Training with artifact data...")
nets_shallow = conv.shallow(epochs=60, data=data_artifact)
error_locations_shallow, erroneous_predictions_shallow = conv.ensemble(nets_shallow)
plt = conv.plot_errors(error_locations_shallow, erroneous_predictions_shallow)
plt.savefig("results/shallow_errors_artifact.png")

