import numpy as np
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K
class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T

    def _get_adaptation_weight(self, t):
        # Dynamic adaptation weight that changes with iterations
        return np.exp(-t / self.T)

    def _weighted_kernel(self, X1, X2, weights, gamma):
        # Apply instance weights to kernel computation
        if X2 is not None:
            K = X1 * weights[:, np.newaxis]
            return sklearn.metrics.pairwise.rbf_kernel(K.T, X2.T, gamma)
        return sklearn.metrics.pairwise.rbf_kernel((X1 * weights[:, np.newaxis]).T, None, gamma)

    def fit_predict(self, Xs, Ys, Xt, Yt):
        list_acc = []
        
        # Normalize and combine data
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        
        # Initialize instance weights
        source_weights = np.ones(ns)
        target_weights = np.ones(nt)
        
        # Class balancing weights
        C = len(np.unique(Ys))
        class_weights = {}
        for c in range(1, C + 1):
            count = np.sum(Ys == c)
            class_weights[c] = np.sqrt(len(Ys) / (count + 1e-6))

        H = np.eye(n) - 1 / n * np.ones((n, n))
        
        M = 0
        Y_tar_pseudo = None
        Xs_new, Xt_new = None, None

        for t in range(self.T):
            # Get dynamic adaptation weight
            adapt_weight = self._get_adaptation_weight(t)
            
            # Update instance weights based on previous predictions
            if Y_tar_pseudo is not None:
                # Update target weights based on prediction confidence
                clf_proba = clf.predict_proba(Xt_new)
                prediction_confidence = np.max(clf_proba, axis=1)
                target_weights = 1 / (1 + np.exp(prediction_confidence))
                target_weights /= np.sum(target_weights)

            # Construct MMD matrix with instance weights
            e_s = source_weights.reshape(-1, 1) / np.sum(source_weights)
            e_t = -target_weights.reshape(-1, 1) / np.sum(target_weights)
            e = np.vstack((e_s, e_t))
            
            M0 = e @ e.T * C * adapt_weight
            
            N = 0
            if Y_tar_pseudo is not None:
                for c in range(1, C + 1):
                    idx_s = Ys == c
                    idx_t = Y_tar_pseudo == c
                    
                    if not np.any(idx_s) or not np.any(idx_t):
                        continue
                    
                    # Create class-specific weight matrix
                    e = np.zeros((n, 1))
                    
                    # Apply class balancing weights
                    e[np.where(idx_s)[0]] = class_weights[c] / (np.sum(idx_s) + 1e-6)
                    e[np.where(idx_t)[0] + ns] = -class_weights[c] / (np.sum(idx_t) + 1e-6)
                    
                    N += e @ e.T

            # Combine marginal and conditional distributions
            M = M0 + (adapt_weight * N)
            M /= np.linalg.norm(M, 'fro')

            # Apply kernel with instance weights
            if self.kernel_type == 'primal':
                K = X
            else:
                instance_weights = np.concatenate([source_weights, target_weights])
                K = self._weighted_kernel(X, None, instance_weights, self.gamma)

            n_eye = m if self.kernel_type == 'primal' else n
            a = K @ M @ K.T + (self.lamb * (1 + t/self.T)) * np.eye(n_eye)
            b = K @ H @ K.T

            # Solve eigenvalue problem
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = A.T @ K
            Z /= np.linalg.norm(Z, axis=0)
            
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            # Use probability-based KNN classifier
            clf = KNeighborsClassifier(n_neighbors=min(5, ns), weights='distance', metric='cosine')
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)

        auc_roc = roc_auc_score(Yt, Y_tar_pseudo, multi_class='ovr')
        f1 = f1_score(Yt, Y_tar_pseudo, average='weighted')
        return acc, Y_tar_pseudo, auc_roc, f1


# class JDA:
#     def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=10):
#         self.kernel_type = kernel_type
#         self.dim = dim
#         self.lamb = lamb
#         self.gamma = gamma
#         self.T = T

#     def fit_predict(self, Xs, Ys, Xt, Yt):
#         list_acc = []
#         X = np.hstack((Xs.T, Xt.T))
#         X /= np.linalg.norm(X, axis=0)
#         m, n = X.shape
#         ns, nt = len(Xs), len(Xt)
#         e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
#         C = len(np.unique(Ys))
#         H = np.eye(n) - 1 / n * np.ones((n, n))

#         M = 0
#         Y_tar_pseudo = None
#         for t in range(self.T):
#             N = 0
#             M0 = e * e.T * C
#             if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
#                 for c in range(1, C + 1):
#                     e = np.zeros((n, 1))
#                     tt = Ys == c
#                     #extra
#                     if np.sum(tt) == 0:
#                       continue 
#                     e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
#                     yy = Y_tar_pseudo == c
#                     ind = np.where(yy == True)
#                     inds = [item + ns for item in ind]
#                     # e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
#                     # e[np.isinf(e)] = 0
#                     if len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)]) > 0:
#                        e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
#                     e[np.isinf(e)] = 0
#                     N = N + np.dot(e, e.T)
#             M = M0 + N
#             M = M / np.linalg.norm(M, 'fro')
#             K = kernel(self.kernel_type, X, None, gamma=self.gamma)
#             n_eye = m if self.kernel_type == 'primal' else n
#             a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
#             # a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
#             w, V = scipy.linalg.eig(a, b)
#             ind = np.argsort(w)
#             A = V[:, ind[:self.dim]]
#             Z = np.dot(A.T, K)
#             Z /= np.linalg.norm(Z, axis=0)
#             Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

#             clf = KNeighborsClassifier(n_neighbors=1)
#             clf.fit(Xs_new, Ys.ravel())
#             Y_tar_pseudo = clf.predict(Xt_new)
#             acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
#             list_acc.append(acc)
#             # print('JDA iteration [{}/{}]: Acc: {:.4f}'.format(t + 1, self.T, acc))

#         auc_roc = roc_auc_score(Yt, Y_tar_pseudo, multi_class='ovr')
#         f1 = f1_score(Yt, Y_tar_pseudo, average='weighted')
#         # f1 = f1_score(Yt, Y_tar_pseudo, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
#         return acc, Y_tar_pseudo , auc_roc , f1
    


