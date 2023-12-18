import numpy as np
from tqdm import tqdm
from ..utils.utils import Hbeta, pca
from multiprocessing import Pool
from functools import partial

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




def x2p(X, distance_matrix=False, tol=1e-5, perplexity=30.0, num_processes=1):
    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    
    if not distance_matrix:
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    else:
        D = X

    P = np.zeros((n, n))
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

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def jedi(X=np.array([]),Z=np.array([]), alpha=0.5, beta=0.5, n_components=2, perplexity=30.0, n_iter=1000, dens=False, verbose=1, random_seed=None, initial_dims=None, num_processes=1):

    if random_seed:
        np.random.seed(random_seed)

    # Check inputs
    if not isinstance(n_components, int):
        print("Error: array X should have type float.")
        return -1

    # Initialize variables
    if initial_dims:
        X = pca(X, initial_dims).real
    (n, d) = X.shape
    
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, n_components)
    dY_KL = np.zeros((n, n_components))
    dY_JS = np.zeros((n, n_components))
    iY = np.zeros((n, n_components))
    gains = np.ones((n, n_components))

    def KL_divergence(P, Q):
        return np.sum(P * np.log(P / Q))


    def cost_function(P, Q, P_prime, alpha, beta):
        JS_alpha_beta = alpha * KL_divergence(P_prime, beta * Q + (1 - beta) * P_prime) + (1 - alpha) * KL_divergence(Q, beta * P_prime + (1 - beta)* Q)
        return KL_divergence(P, Q) - JS_alpha_beta


    def KL_divergence_grad_update(P, Q): 
        PQ = P - Q
        for i in range(n):
            dY_KL[i, :] = (PQ[i, :] * num[i, :]) @ (Y[i, :] - Y)
            
    
    def JS_alpha_beta_grad_update(P, Q, alpha, beta, num): 
        common_vec1 = P*Q / (beta*Q + (1-beta)*P)
        common_vec1[range(n), range(n)] = 0
        common_vec1 = alpha*beta*common_vec1.sum()

        common_vec2 = Q*(1 + np.log(Q) - (1-beta)*Q / (beta*P + (1-beta)*Q) - np.log(beta*P + (1-beta)*Q))
        common_vec2[range(n), range(n)] = 0
        common_vec2 = (1-alpha) * common_vec2.sum()

        for i in range(n):
            vec_before_brackets = num[i, :] * Q[i, :] 
            vec1 = alpha * beta * P[i, :] / (beta * Q[i, :] + (1 - beta) * P[i, :])
            vec2 = (1-alpha)*(-1-np.log(Q[i, :]) + (1-beta)*Q[i, :]/(beta*P[i, :]+(1-beta)*Q[i, :]) + np.log(beta*P[i, :]+(1-beta)*Q[i, :]))
            vec = vec_before_brackets * (vec1 - common_vec1 + vec2 + common_vec2)   
            dY_JS[i, :] = vec @ (Y[i, :] - Y) # sum over j
        

    # Compute matrix P
    P = x2p(X, distance_matrix=False, tol=1e-5, perplexity=perplexity, num_processes=num_processes)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    # Early exaggeration
    P = P * 4.			
    P = np.maximum(P, 1e-12)

    # Compute matrix P_prime 
    P_prime = x2p(Z, distance_matrix=True, tol=1e-5, perplexity=perplexity, num_processes=num_processes)
    P_prime = P_prime + np.transpose(P_prime)
    P_prime = P_prime / np.sum(P_prime)
    # Early exaggeration
    P_prime = P_prime * 4.			
    P_prime = np.maximum(P_prime, 1e-12)
    
    C_old = np.sum(P)+100
    # Run iterations
    print('Performing optimization...')

    @jit(nopython=True)
    def optim(P_prime, Q, alpha, beta, n_iter, Y, P, n, 
              initial_momentum, final_momentum, gains, 
              iY, min_gain, eta, verbose, C_old):
        for iter in range(n_iter):

            # Compute pairwise affinities
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            # Compute gradients
            KL_divergence_grad_update(P, Q)
            JS_alpha_beta_grad_update(P_prime, Q, alpha, beta, num)

            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * (((dY_KL+dY_JS) > 0.) != (iY > 0.)) + \
                    (gains * 0.8) * (((dY_KL+dY_JS) > 0.) == (iY > 0.))
            gains = np.clip(gains, min_gain, None)
            iY = momentum * iY - eta * (gains * (dY_KL+dY_JS))
            Y = Y + iY
            m = np.empty((2, Y.shape[0]))
            m[0] = np.mean(Y, 0)
            m[1] = np.mean(Y, 0)
            Y = Y - m

            # Compute current value of cost function
            if (iter + 1) % 50 == 0 and verbose > 1:
                C = np.sum(P * np.log(P / Q))
                if abs(C-C_old)<5e-5 and iter > 100:
                    # print("Early stopping due to small change", C, C_old)
                    break
                C_old = C
                # print("Iteration %d: error is %f" % (iter + 1, C))

            # Stop lying about P-values
            if iter == 100:
                P = P / 4.
        return Y
    Y = optim(P_prime, Q, alpha, beta, n_iter, Y, P, n, 
              initial_momentum, final_momentum, gains, 
              iY, min_gain, eta, verbose, C_old)
    # Return solution
    return Y
