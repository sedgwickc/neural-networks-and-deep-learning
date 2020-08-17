# -----------------------------------------------------------------------------
# artifact_experiments.py
# This script contains the experiments used to test the ability of a basic ANN
# and deep learning ANN described by Nielsen. It also generates graphs of the
# performance of the best out of 3 networks trained of each type.


import src.network3 as network3
import src.conv as conv

data_artifact_exp = "../data/mnist_expanded_25_25.pkl.gz"
mini_batch_size = 10

print("Running deep learning network experiments with just artifact training data...")

# experiments run by Nielson to achieve %99.67 accuracy
nets_train = conv.double_fc_dropout(0.5, 0.5, 0.5, 5, data_artifact_exp)
# plot the erroneous digits in the ensemble of nets just trained
error_locations, erroneous_predictions = conv.ensemble(nets_train)
plt = conv.plot_errors(error_locations, erroneous_predictions)
plt.savefig("results/ensemble_errors_artifact_train.png")
# plot the filters learned by the first of the nets just trained
plt = conv.plot_filters(nets_train[0], 0, 5, 4)
plt.savefig("results/net_full_layer_0_artifact_train.png")
plt = conv.plot_filters(nets_train[0], 1, 8, 5)
plt.savefig("results/net_full_layer_1_artifact_train.png")

