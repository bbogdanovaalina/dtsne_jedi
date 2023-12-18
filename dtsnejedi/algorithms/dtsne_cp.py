from tqdm import tqdm
from ..utils.utils_cp import Hbeta_cp, pca_cp
import matplotlib.pyplot as plt
import cupy as cp
from multiprocessing import Pool
from functools import partial


def BetaDens_cp(D=cp.array([]), beta=cp.array([]), i=None):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """
    
    # Compute P-row and corresponding perplexity
    n = D.shape[0]
    #symm_beta = 1. / ((1. / np.sqrt(beta[np.concatenate((np.r_[0:i], np.r_[i+1:n+1]))] / 2.) + 1. / np.sqrt(beta[i] / 2.)) / 2.)**2 / 2.
    symm_beta = 4 * beta * beta[i] / (2 * cp.sqrt(beta * beta[i]) + beta[i] + beta)
    gamma = symm_beta.copy()
    #print(beta[i], symm_beta[i], gamma[i], beta.mean(), symm_beta.mean(), gamma.mean())
    P = cp.exp(-D.copy() * symm_beta[cp.concatenate((cp.r_[0:i], cp.r_[i+1:n+1])),0])
    sumP = max(sum(P), cp.finfo(cp.double).eps)
    P = P / sumP
    return P, gamma[cp.concatenate((cp.r_[0:i], cp.r_[i+1:n+1])),0]



def pairwise_dist(i, D=None, logU=None, tol=None, n=None):
    # Compute the Gaussian kernel and entropy for the current precision
    print('Pdist for one point')
    betamin = -cp.inf
    betamax = cp.inf
    beta_i = 1

    concated = cp.concatenate((cp.arange(0, i), cp.arange(i+1, n)))
    Di = D[i, concated]
    (H, thisP) = Hbeta_cp(Di, beta_i)
    # Evaluate whether the perplexity is within tolerance
    Hdiff = H - logU
    tries = 0
    while cp.abs(Hdiff) > tol and tries < 50:

        # If not, increase or decrease precision
        if Hdiff > 0:
            betamin = beta_i.copy()
            if betamax == cp.inf or betamax == -cp.inf:
                beta_i = beta_i * 2.
            else:
                beta_i = (beta_i + betamax) / 2.
        else:
            betamax = beta_i.copy()
            if betamin == cp.inf or betamin == -cp.inf:
                beta_i = beta_i / 2.
            else:
                beta_i = (beta_i + betamin) / 2.

        # Recompute the values
        (H, thisP) = Hbeta_cp(Di, beta_i)
        Hdiff = H - logU
        tries += 1
    print('Dist for one point calced')
    return thisP, beta_i, i


def x2p_cp(X=cp.array([]), tol=1e-5, perplexity=30.0, dens=False, num_processes=2):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = cp.sum(cp.square(X), 1)
    D = cp.add(cp.add(-2 * cp.dot(X, X.T), sum_X).T, sum_X)
    P = cp.zeros((n, n))
    gamma = cp.zeros((n, n))
    beta = cp.ones((n, 1))
    logU = cp.log(perplexity)



    # Loop over all datapoints
    indices = cp.arange(n)
    calc_fn = partial(pairwise_dist, D=D, logU=logU, tol=tol, n=n)
    print('Start calculating pairwise dists')
    with Pool(num_processes) as p:
        P = list(tqdm(p.imap(calc_fn, indices), total=len(indices)))
    print('Finish calcs')

    # for i in tqdm(range(n)):
    #     thisP, beta[i], i = pairwise_dist(i, D, logU, tol, n, )
        # P[i, cp.concatenate((cp.arange(0, i), cp.arange(i+1, n)))] = thisP
    if dens:
        for i in range(n):
            Di = D[i, cp.concatenate((cp.arange(0, i), cp.arange(i+1, n)))]
            thisP, thisgamma = BetaDens_cp(Di, beta, i)
            P[i, cp.concatenate((cp.arange(0, i), cp.arange(i+1, n)))] = thisP
            gamma[i, cp.concatenate((cp.arange(0, i), cp.arange(i+1, n)))] = thisgamma
        gamma /= gamma.max()
    # Return final P-matrix
    print("Mean value of sigma: %f" % cp.mean(cp.sqrt(1 / beta)))
    return P, gamma




def dtsne_cp(X=cp.array([]), n_components=2, perplexity=30.0, n_iter=1000, dens=False, verbose=1, random_seed=None, initial_dims=None):
    
    if random_seed:
        cp.random.seed(random_seed)
    # Check inputs
    if not isinstance(n_components, int):
        print("Error: array X should have type float.")
        return -1
    if dens:
        print("Runs dtSNE variant of algorithm.")
    # Initialize variables
    if initial_dims:
        X = pca_cp(X, initial_dims).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = pca_cp(X, n_components).real
    Y = 1e-4 * Y / cp.std(Y)
    dY = cp.zeros((n, n_components))
    iY = cp.zeros((n, n_components))
    gains = cp.ones((n, n_components))

    # Compute P-values
    P, gamma = x2p_cp(X, 1e-5, perplexity, dens)
    P = P + cp.transpose(P)
    P = P / cp.sum(P)
    # Early exaggeration
    P = P * 4.  
    P = cp.maximum(P, cp.finfo(cp.double).eps)
   
    C_old = cp.sum(P)+100
    # Run iterations
    print('Performing optimization...')
    for iter in tqdm(range(n_iter)):

        # Compute pairwise affinities
        sum_Y = cp.sum(cp.square(Y), 1)
        num = -2. * cp.dot(Y, Y.T)
        if dens:
            num = 1. / (1. + gamma * cp.add(cp.add(num, sum_Y).T, sum_Y))
        else:
            num = 1. / (1. + cp.add(cp.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / cp.sum(num)
        Q = cp.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        if dens:
            for i in range(n):
                dY[i, :] = 4 * cp.sum(cp.tile(PQ[:, i] * num[:, i] * gamma[:, i], (n_components, 1)).T * (Y[i, :] - Y) , 0)
        else:
            for i in range(n):
                dY[i, :] = cp.sum(cp.tile(PQ[:, i] * num[:, i], (n_components, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        #Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        
        if (iter + 1) % 50 == 0 and verbose > 1:
            C = cp.sum(P * cp.log(P / Q))
            if abs(C-C_old)<5e-5 and iter > 100:
                print("Early stopping due to small change", C, C_old)
                break
            C_old = C
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    
    # Return solution
    return Y
