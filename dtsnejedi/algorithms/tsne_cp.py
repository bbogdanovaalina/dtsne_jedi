import cupy as cp
# from categorical_scatter import categorical_scatter_2d


# from scipy.stats import ortho_group

def rvs(dim=3):
     random_state = cp.random
     H = cp.eye(dim)
     D = cp.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = cp.sign(x[0])
         x[0] -= D[n-1]*cp.sqrt((x*x).sum())
         # Householder transformation
         Hx = (cp.eye(dim-n+1) - 2.*cp.outer(x, x)/(x*x).sum())
         mat = cp.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = cp.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to cp.dot(cp.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H

def get_gaussian_data_cp(dims=2, n_clusters=5, max_points=200):
    X = cp.empty((0, dims))
    y = cp.empty((0, 1, ))

    for i in range(n_clusters):
        num_samples = int(cp.random.randint(10, max_points))
        means = 100 * (cp.random.rand(dims) - 0.5)
        diag = cp.abs(cp.diag(40 * cp.random.rand(dims)))
        O = rvs(dims)
        covs = O.T @ diag @ O
        cloud = cp.random.multivariate_normal(means, covs, num_samples)

        X = cp.concatenate((X, cloud))
        y = cp.vstack((y, cp.ones(shape=(num_samples, 1, )) * i))
    return X, y

def neg_squared_euc_dists_cp(X):
    sum_X = cp.sum(cp.square(X), 1)
    D = cp.add(cp.add(-2 * cp.dot(X, X.T), sum_X).T, sum_X)
    return -D


def softmax_cp(X, diag_zero=True, zero_index=None):
    e_x = cp.exp(X - cp.max(X, axis=1).reshape([-1, 1]))

    if zero_index is None:
        if diag_zero:
            cp.fill_diagonal(e_x, 0.)
    else:
        e_x[:, zero_index] = 0.

    e_x = e_x + 1e-8  # numerical stability
    return e_x / e_x.sum(axis=1).reshape([-1, 1])


def calc_prob_matrix_cp(distances, sigmas=None, zero_index=None):
    """Convert a distances matrix to a matrix of probabilities."""
    if sigmas is not None:
        two_sig_sq = 2. * cp.square(sigmas.reshape((-1, 1)))
        return softmax_cp(distances / two_sig_sq, zero_index=zero_index)
    else:
        return softmax_cp(distances, zero_index=zero_index)


def binary_search_cp(eval_fn, target, tol=1e-10, max_iter=10000,
                  lower=1e-20, upper=1000.):
    
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if cp.abs(val - target) <= tol:
            break
    return guess


def calc_perplexity_cp(prob_matrix):
    entropy = -cp.sum(prob_matrix * cp.log2(prob_matrix), 1)
    perplexity = 2 ** entropy
    return perplexity


def perplexity_cp(distances, sigmas, zero_index):
    return calc_perplexity_cp(
        calc_prob_matrix_cp(distances, sigmas, zero_index))


def find_optimal_sigmas_cp(distances, target_perplexity):
    sigmas = []
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: \
            perplexity_cp(distances[i:i+1, :], cp.array(sigma), i)
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search_cp(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return cp.array(sigmas)


def p_conditional_to_joint_cp(P):
    """Given conditional probabilities matrix P, return
    approximation of joint distribution probabilities."""
    return (P + P.T) / (2. * P.shape[0])


def q_joint_cp(Y):
    """Given low-dimensional representations Y, compute
    matrix of joint probabilities with entries q_ij."""
    # Get the distances from every point to every other
    distances = neg_squared_euc_dists_cp(Y)
    # Take the elementwise exponent
    exp_distances = cp.exp(distances)
    # Fill diagonal with zeroes so q_ii = 0
    cp.fill_diagonal(exp_distances, 0.)
    # Divide by the sum of the entire exponentiated matrix
    return exp_distances / cp.sum(exp_distances), None


def symmetric_sne_grad_cp(P, Q, Y, _):
    """Estimate the gradient of the cost with respect to Y"""
    pq_diff = P - Q  # NxN matrix
    pq_expanded = cp.expand_dims(pq_diff, 2)  #NxNx1
    y_diffs = cp.expand_dims(Y, 1) - cp.expand_dims(Y, 0)  #NxNx2
    grad = 4. * (pq_expanded * y_diffs).sum(1)  #Nx2
    return grad


def q_tsne_cp(Y):
    """t-SNE: Given low-dimensional representations Y, compute
    matrix of joint probabilities with entries q_ij."""
    distances = neg_squared_euc_dists_cp(Y)
    inv_distances = cp.power(1. - distances, -1)
    cp.fill_diagonal(inv_distances, 0.)
    return inv_distances / cp.sum(inv_distances), inv_distances


def tsne_grad_cp(P, Q, Y, distances):
    """t-SNE: Estimate the gradient of the cost with respect to Y."""
    pq_diff = P - Q  # NxN matrix
    pq_expanded = cp.expand_dims(pq_diff, 2)  # NxNx1
    y_diffs = cp.expand_dims(Y, 1) - cp.expand_dims(Y, 0)  # NxNx2
    # Expand our distances matrix so can multiply by y_diffs
    distances_expanded = cp.expand_dims(distances, 2)  # NxNx1
    # Weight this (NxNx2) by distances matrix (NxNx1)
    y_diffs_wt = y_diffs * distances_expanded  # NxNx2
    grad = 4. * (pq_expanded * y_diffs_wt).sum(1)  # Nx2
    return grad


def p_join_cpt(X, target_perplexity):
    # Get the negative euclidian distances matrix for our data
    distances = neg_squared_euc_dists_cp(X)
    # Find optimal sigma for each row of this distances matrix
    sigmas = find_optimal_sigmas_cp(distances, target_perplexity)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix_cp(distances, sigmas)
    # Go from conditional to joint probabilities matrix
    P = p_conditional_to_joint_cp(p_conditional)
    return P


def estimate_sne_cp(X, y, P, rng, num_iters, q_fn, grad_fn, learning_rate,
                 momentum, plot):
    
    # Initialise our 2D representation
    Y = rng.normal(0., 0.0001, [X.shape[0], 2])

    # Initialise past values (used for momentum)
    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

    # Start gradient descent loop
    for i in range(num_iters):

        # Get Q and distances (distances only used for t-SNE)
        Q, distances = q_fn(Y)
        # Estimate gradients with respect to Y
        grads = grad_fn(P, Q, Y, distances)

        # Update Y
        Y = Y - learning_rate * grads
        if momentum:  # Add momentum
            Y += momentum * (Y_m1 - Y_m2)
            # Update previous Y's for momentum
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()

        # Plot sometimes
        # if plot and i % (num_iters / plot) == 0:
        #     categorical_scatter_2d(Y, y, alpha=1.0, ms=6,
        #                            show=True, figsize=(9, 6))

    return Y