import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import OneHotEncoder


def build_similarity_graph(X, var=1.0, eps=0.0, k=0, knn_or=True):
    n = X.shape[0]
    W = np.zeros((n, n))

    similarities = np.exp(- (euclidean_distances(X) ** 2) / (2 * var))
    similarities[np.arange(n), np.arange(n)] = 0

    if k == 0:
        W = similarities
        W[W < eps] = 0

    elif k != 0:
        for i in range(n):
            k_neighbors_indices = np.argsort(-similarities[i])[:k]

            if knn_or:
                for j in k_neighbors_indices:
                    W[i, j] = similarities[i, j]
                    W[j, i] = similarities[i, j]

            else:
                for j in k_neighbors_indices:
                    if i in np.argsort(-similarities[j])[:k]:
                        W[i, j] = similarities[i, j]
                        W[j, i] = similarities[i, j]
            
    return W


def build_laplacian(W, laplacian_normalization='unn'):
    D = np.diag(W.sum(axis=1))
    L = D - W

    if laplacian_normalization == "sym":
        D_inv_sqrt = np.sqrt(np.linalg.pinv(D))
        L = D_inv_sqrt.dot(L).dot(D_inv_sqrt)

    elif laplacian_normalization == "rw":
        D_inv = np.linalg.pinv(D)
        L = D_inv.dot(L)

    return L


def build_laplacian_regularized(X, laplacian_regularization=1.0, var=1.0, eps=0.0, k=0, laplacian_normalization=""):
    W = build_similarity_graph(X, var, eps, k)
    L = build_laplacian(W, laplacian_normalization)
    Q = L + laplacian_regularization*np.eye(W.shape[0])
    return Q


def mask_labels(Y, l, per_class=False):
    num_samples = np.size(Y, 0)
    min_label = Y.min()
    max_label = Y.max()
    assert min_label == 1

    if not per_class:
        Y_masked = np.zeros(num_samples)
        indices_to_reveal = np.arange(num_samples)
        np.random.shuffle(indices_to_reveal)
        indices_to_reveal = indices_to_reveal[:l]
        Y_masked[indices_to_reveal] = Y[indices_to_reveal]
    else:
        Y_masked = np.zeros(num_samples)
        for label in range(min_label, max_label+1):
            indices = np.where( Y == label)[0]
            np.random.shuffle(indices)
            indices = indices[:l]
            Y_masked[indices] = Y[indices]

    return Y_masked


def compute_hfs(L, Y, soft=False, c_l=0.99, c_u=0.01):
    num_samples = L.shape[0]
    l_idx = np.where(Y != 0)[0]
    u_idx = np.where(Y == 0)[0]
    y = OneHotEncoder().fit_transform(Y.reshape(-1, 1)).toarray()[:, 1:]

    if not soft:    
        f_l = y[l_idx]
        L_uu = L[u_idx][:, u_idx]
        L_ul = L[u_idx][:, l_idx]
        f_u = - (np.linalg.pinv(L_uu)).dot(L_ul.dot(f_l))
        f = np.zeros_like(y)
        f[l_idx] = f_l
        f[u_idx] = f_u

    else:
        C = np.zeros(num_samples)
        C[l_idx] = c_l
        C[u_idx] = c_u
        C = np.diag(C)
        f = np.linalg.pinv(((np.linalg.pinv(C)).dot(L) + np.identity(num_samples))).dot(y)

    labels = 1 + f.argmax(axis=1)
    return labels, f
