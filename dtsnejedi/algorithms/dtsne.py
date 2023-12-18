
import numpy as np
from tqdm import tqdm
from ..utils.utils import Hbeta, pca
from multiprocessing import Pool
from functools import partial
from numba import jit


def BetaDens(D=np.array([]), beta=np.array([]), i=None):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    n = D.shape[0]
    #symm_beta = 1. / ((1. / np.sqrt(beta[np.concatenate((np.r_[0:i], np.r_[i+1:n+1]))] / 2.) + 1. / np.sqrt(beta[i] / 2.)) / 2.)**2 / 2.
    symm_beta = 4 * beta * beta[i] / (2 * np.sqrt(beta * beta[i]) + beta[i] + beta)
    gamma = symm_beta.copy()

    P = np.exp(-D.copy() * symm_beta[np.concatenate((np.r_[0:i], np.r_[i+1:n+1])),0])
    sumP = max(sum(P),np.finfo(np.double).eps)
    P = P / sumP
    return P, gamma[np.concatenate((np.r_[0:i], np.r_[i+1:n+1])),0]


def pairwise_dist(i, D=None, logU=None, tol=None, n=None):
    # Compute the Gaussian kernel and entropy for the current precision
    # print('Pdist for one point')
    betamin = -np.inf
    betamax = np.inf
    beta_i = 1

    concated = np.concatenate((np.arange(0, i), np.arange(i+1, n)))
    Di = D[i, concated]
    (H, thisP) = Hbeta(Di, beta_i)
    # Evaluate whether the perplexity is within tolerance
    Hdiff = H - logU
    tries = 0
    while np.abs(Hdiff) > tol and tries < 50:

        # If not, increase or decrease precision
        if Hdiff > 0:
            betamin = beta_i
            if betamax == np.inf or betamax == -np.inf:
                beta_i = beta_i * 2.
            else:
                beta_i = (beta_i + betamax) / 2.
        else:
            betamax = beta_i
            if betamin == np.inf or betamin == -np.inf:
                beta_i = beta_i / 2.
            else:
                beta_i = (beta_i + betamin) / 2.

        # Recompute the values
        (H, thisP) = Hbeta(Di, beta_i)
        Hdiff = H - logU
        tries += 1
    # print('Dist for one point calced')
    thisP = np.insert(thisP, i, 0)
    return thisP, beta_i, i


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0, dens=False, num_processes=1):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    gamma = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    indices = np.arange(n)
    calc_fn = partial(pairwise_dist, D=D, logU=logU, tol=tol, n=n)
    print('Start calculating pairwise dists')
    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(calc_fn, indices), total=len(indices)))
    print('Finish calcs')

    for thisP, beta_i, i in res:
        P[i] = thisP
        beta[i] = beta_i
    
    if dens:
        for i in range(n):
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            thisP, thisgamma = BetaDens(Di, beta, i)
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
            gamma[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisgamma
        gamma /= gamma.max()
    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P, gamma


@jit(nopython=True)
def optim(n_iter, Y, gamma, P, dens, n, dY, n_components, 
              initial_momentum, final_momentum, gains, 
              iY, min_gain, eta, verbose, C_old):
    for iter in range(n_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        if dens:
            num = 1. / (1. + gamma * np.add(np.add(num, sum_Y).T, sum_Y))
        else:
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))

        # num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        if dens:
            for i in range(n):
                m = np.empty((2, PQ.shape[0]))
                m[0] = PQ[:, i] * num[:, i] * gamma[:, i]
                m[1] = PQ[:, i] * num[:, i] * gamma[:, i]
                dY[i, :] = 4 * np.sum(m.T * (Y[i, :] - Y) , 0)
                # dY[i, :] = 4 * np.sum(np.tile(PQ[:, i] * num[:, i] * gamma[:, i], (n_components, 1)).T * (Y[i, :] - Y) , 0)
        else:
            for i in range(n):
                m = np.empty((2, PQ.shape[0]))
                m[0] = PQ[:, i] * num[:, i]
                m[1] = PQ[:, i] * num[:, i]               
                # m = np.array([PQ[:, i] * num[:, i], PQ[:, i] * num[:, i]])
                # dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (n_components, 1)).T * (Y[i, :] - Y), 0)
                dY[i, :] = np.sum(m.T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains = np.clip(gains, min_gain, None) #gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        #Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        
        if (iter + 1) % 50 == 0 and verbose > 1:
            C = np.sum(P * np.log(P / Q))
            if abs(C-C_old)<5e-5 and iter > 100:
                print("Early stopping due to small change", C, C_old)
                break
            C_old = C
            # print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    return Y
    


def dtsne(X=np.array([]), n_components=2, perplexity=30.0, n_iter=1000, dens=False, verbose=1, random_seed=None, initial_dims=None, num_processes=1):
    
    if random_seed:
        np.random.seed(random_seed)
    # Check inputs
    if not isinstance(n_components, int):
        print("Error: array X should have type float.")
        return -1
    if dens:
        print("Runs dtSNE variant of algorithm.")
    # Initialize variables
    if initial_dims:
        X = pca(X, initial_dims).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = pca(X, n_components).real
    Y = 1e-4 * Y / np.std(Y)
    dY = np.zeros((n, n_components))
    iY = np.zeros((n, n_components))
    gains = np.ones((n, n_components))

    # Compute P-values
    P, gamma = x2p(X, 1e-5, perplexity, dens, num_processes=num_processes)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    # Early exaggeration
    P = P * 4.  
    P = np.maximum(P, np.finfo(np.double).eps)
    
    C_old = np.sum(P)+100
    # Run iterations
    print('Performing optimization...')

    Y = optim(n_iter, Y, gamma, P, dens, n, dY, n_components, 
            initial_momentum, final_momentum, gains, 
            iY, min_gain, eta, verbose, C_old)
    
    # Return solution
    return Y


def get_gaussian_data(dims=2, n_clusters=5):
    X = []
    y = []
    from scipy.stats import ortho_group

    for cluster in range(n_clusters):
        means = 100 * (np.random.rand(dims) - 0.5)
        diag = np.abs(np.diag(40 * np.random.rand(dims)))
        O = ortho_group.rvs(dims)
        covs = O.T @ diag @ O
        num_samples = np.random.randint(10, 200)

        cloud = np.random.multivariate_normal(means, covs, num_samples)

        X.append(cloud)
        y.append([cluster] * num_samples)

    return np.concatenate(X), np.concatenate(y)
    
