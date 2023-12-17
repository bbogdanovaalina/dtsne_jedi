from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# from load_data import load_mnist
from tsne import estimate_sne, tsne_grad, symmetric_sne_grad, q_tsne, q_joint, p_joint
from tsne_cp import estimate_sne_cp, tsne_grad_cp, symmetric_sne_grad_cp, q_tsne_cp, q_joint_cp, p_joint_cp

from tsne_cp import get_gaussian_data_cp


# Set global parameters
NUM_POINTS = 200            # Number of samples from MNIST
CLASSES_TO_USE = [0, 1, 8]  # MNIST classes to use
PERPLEXITY = 20
SEED = 1                    # Random seed
MOMENTUM = 0.9
LEARNING_RATE = 10.
NUM_ITERS = 500             # Num iterations to train for
TSNE = True                # If False, Symmetric SNE
NUM_PLOTS = 0              # Num. times to plot in training


def main():
    # numpy RandomState for reproducibility
    rng = np.random.RandomState(SEED)

    # Load the first NUM_POINTS 0's, 1's and 8's from MNIST
    X, y = get_gaussian_data_cp(2, 6, 50)

    # Obtain matrix of joint probabilities p_ij
    P = p_joint(X, PERPLEXITY)

    # Fit SNE or t-SNE
    Y = estimate_sne(X, y, P, rng,
                     num_iters=NUM_ITERS,
                     q_fn=q_tsne if TSNE else q_joint,
                     grad_fn=tsne_grad if TSNE else symmetric_sne_grad,
                     learning_rate=LEARNING_RATE,
                     momentum=MOMENTUM,
                     plot=NUM_PLOTS)
    print(Y.shape)


main()
