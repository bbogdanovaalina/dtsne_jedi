import cupy as cp

def pca_cp(X=cp.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to get initiall embedding.
    """

    print("Getting initial embedding using PCA...")
    (n, d) = X.shape
    X = X - cp.tile(cp.mean(X, 0), (n, 1))
    (l, M) = cp.linalg.eigh(cp.dot(X.T, X))
    Y = cp.dot(X, M[:, 0:no_dims])
    return Y

def Hbeta_cp(D=cp.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = cp.exp(-D.copy() * beta) 
    sumP = max(sum(P), cp.finfo(cp.double).eps)
    H = cp.log(sumP) + beta * cp.sum(D * P) / sumP
    P = P / sumP
    return H, P
