import cupy as cp

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


def get_gaussian_data_jedi_cp(dims=2, n_clusters=5, max_points=200):
    X = cp.empty((0, dims))
    y = cp.empty((0, 1, ))
    cluster_size = []

    for i in range(n_clusters):
        num_samples = int(cp.random.randint(10, max_points))
        means = 100 * (cp.random.rand(dims) - 0.5)
        diag = cp.abs(cp.diag(40 * cp.random.rand(dims)))
        O = rvs(dims)
        covs = O.T @ diag @ O
        cloud = cp.random.multivariate_normal(means, covs, num_samples)

        X = cp.concatenate((X, cloud))
        y = cp.vstack((y, cp.ones(shape=(num_samples, 1, )) * i))
        cluster_size.append(num_samples)
    return X, y, cluster_size
