import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def construct_mmd_matrix(ns, nt):
    """Construct MMD matrix"""
    W = np.zeros((ns + nt, ns + nt))
    W[:ns, :ns] = 1.0 / (ns * ns)
    W[ns:, ns:] = 1.0 / (nt * nt)
    W[:ns, ns:] = -1.0 / (ns * nt)
    W[ns:, :ns] = -1.0 / (ns * nt)
    return W

def calculate_graph_laplacian(X, y, C):
    """Calculate graph Laplacian for target data"""
    n = X.shape[0]
    W = np.zeros((n, n))
    
    for c in range(C):
        idx = (y == c)
        if np.sum(idx) > 0:
            Xc = X[idx]
            dist = rbf_kernel(Xc)
            W[np.ix_(idx, idx)] = dist
            
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L

class DTLC:
    def __init__(self, dim=10, max_iter=10, alpha=1.0, beta=1.0, eta=1.0):
        self.dim = dim
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

    def solve_generalized_eigen(self, X, W, M):
        """
        Solve the generalized eigendecomposition problem
        X: features matrix (samples × features)
        """
        XWXt = np.dot(np.dot(X.T, W), X)  # (m × m)
        XMXt = np.dot(np.dot(X.T, M), X) + np.eye(X.shape[1]) * 1e-6  # (m × m)

        eigenvals, eigenvecs = eigh(XWXt, XMXt)

        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        P = eigenvecs[:, :self.dim]
        return P

    def fit_predict(self, Xs, Ys, Xt, Yt):
        Xs = np.array(Xs)
        Xt = np.array(Xt)
        Ys = np.array(Ys)
        Yt = np.array(Yt)

        ns = Xs.shape[0]
        nt = Xt.shape[0]
        C = len(np.unique(Ys))

        X = np.vstack((Xs, Xt))
        X = X / (np.linalg.norm(X, axis=1).reshape(-1, 1) + 1e-6)

        W = construct_mmd_matrix(ns, nt)
        M = np.zeros((ns + nt, ns + nt))
        prev_yt_pred = None

        for t in range(self.max_iter):
            P = self.solve_generalized_eigen(X, W, M)

            Zs = np.dot(Xs, P)
            Zt = np.dot(Xt, P)

            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Zs, Ys.ravel())
            yt_pred = clf.predict(Zt)

            L = calculate_graph_laplacian(Zt, yt_pred, C)

            W = construct_mmd_matrix(ns, nt)
            M = np.block([
                [np.zeros((ns, ns)), np.zeros((ns, nt))],
                [np.zeros((nt, ns)), L]
            ])

            if prev_yt_pred is not None and np.mean(yt_pred == prev_yt_pred) > 0.99:
                break
            prev_yt_pred = yt_pred.copy()

        acc = accuracy_score(Yt, yt_pred)
        try:
            auc_roc = roc_auc_score(Yt, yt_pred, multi_class='ovr')
        except ValueError:
            auc_roc = roc_auc_score(Yt, yt_pred)
        f1 = f1_score(Yt, yt_pred, average='weighted')

        return acc, yt_pred, auc_roc, f1

    def fit(self, Xs, Xt):
        Xs = np.array(Xs)
        Xt = np.array(Xt)

        ns = Xs.shape[0]
        nt = Xt.shape[0]

        X = np.vstack((Xs, Xt))
        X = X / (np.linalg.norm(X, axis=1).reshape(-1, 1) + 1e-6)

        W = construct_mmd_matrix(ns, nt)
        M = np.zeros((ns + nt, ns + nt))

        P = self.solve_generalized_eigen(X, W, M)

        Xs_new = np.dot(Xs, P)
        Xt_new = np.dot(Xt, P)

        return Xs_new, Xt_new
