import cupy as cp
from tqdm import tqdm
from utils_cp import Hbeta_cp, pca_cp

def x2p(X, distance_matrix=False, tol=1e-5, perplexity=30.0):
    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    
    if not distance_matrix:
        sum_X = cp.sum(cp.square(X), 1)
        D = cp.add(cp.add(-2 * cp.dot(X, X.T), sum_X).T, sum_X)
    else:
        D = X

    P = cp.zeros((n, n))
    beta = cp.ones((n, 1))
    logU = cp.log(perplexity)

    # Loop over all datapoints 
    for i in tqdm(range(n)):

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -cp.inf
        betamax = cp.inf
        Di = D[i, cp.concatenate((cp.r_[0:i], cp.r_[i+1:n]))]
        (H, thisP) = Hbeta_cp(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while cp.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == cp.inf or betamax == -cp.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == cp.inf or betamin == -cp.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_cp(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, cp.concatenate((cp.r_[0:i], cp.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % cp.mean(cp.sqrt(1 / beta)))
    return P


def jedi(X=cp.array([]),Z=cp.array([]), alpha=0.5, beta=0.5, n_components=2, perplexity=30.0, n_iter=1000, dens=False, verbose=1, random_seed=None, initial_dims=None):

    if random_seed:
        cp.random.seed(random_seed)

    # Check inputs
    if not isinstance(n_components, int):
        print("Error: array X should have type float.")
        return -1

    # Initialize variables
    if initial_dims:
        X = pca_cp(X, initial_dims).real
    (n, d) = X.shape
    
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = cp.random.randn(n, n_components)
    dY_KL = cp.zeros((n, n_components))
    dY_JS = cp.zeros((n, n_components))
    iY = cp.zeros((n, n_components))
    gains = cp.ones((n, n_components))

    def KL_divergence(P, Q):
        return cp.sum(P * cp.log(P / Q))


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

        common_vec2 = Q*(1 + cp.log(Q) - (1-beta)*Q / (beta*P + (1-beta)*Q) - cp.log(beta*P + (1-beta)*Q))
        common_vec2[range(n), range(n)] = 0
        common_vec2 = (1-alpha) * common_vec2.sum()

        for i in range(n):
            vec_before_brackets = num[i, :] * Q[i, :] 
            vec1 = alpha * beta * P[i, :] / (beta * Q[i, :] + (1 - beta) * P[i, :])
            vec2 = (1-alpha)*(-1-cp.log(Q[i, :]) + (1-beta)*Q[i, :]/(beta*P[i, :]+(1-beta)*Q[i, :]) + cp.log(beta*P[i, :]+(1-beta)*Q[i, :]))
            vec = vec_before_brackets * (vec1 - common_vec1 + vec2 + common_vec2)   
            dY_JS[i, :] = vec @ (Y[i, :] - Y) # sum over j
        

    # Compute matrix P
    P = x2p(X, distance_matrix=False, tol=1e-5, perplexity=perplexity)
    P = P + cp.transpose(P)
    P = P / cp.sum(P)
    # Early exaggeration
    P = P * 4.			
    P = cp.maximum(P, 1e-12)

    # Compute matrix P_prime 
    P_prime = x2p(Z, distance_matrix=True, tol=1e-5, perplexity=perplexity)
    P_prime = P_prime + cp.transpose(P_prime)
    P_prime = P_prime / cp.sum(P_prime)
    # Early exaggeration
    P_prime = P_prime * 4.			
    P_prime = cp.maximum(P_prime, 1e-12)
    
    C_old = cp.sum(P)+100
    # Run iterations
    print('Performing optimization...')
    for iter in tqdm(range(n_iter)):

        # Compute pairwise affinities
        sum_Y = cp.sum(cp.square(Y), 1)
        num = -2. * cp.dot(Y, Y.T)
        num = 1. / (1. + cp.add(cp.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / cp.sum(num)
        Q = cp.maximum(Q, 1e-12)

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
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * (dY_KL+dY_JS))
        Y = Y + iY
        Y = Y - cp.tile(cp.mean(Y, 0), (n, 1))

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
