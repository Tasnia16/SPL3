import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch as th
from utils import _get_first_singular_vectors_power_method, _get_first_singular_vectors_svd, _svd_flip_1d



def compute_laplacian(X, k=5):
    """Compute the Laplacian matrix for input data X."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    W = np.zeros((X.shape[0], X.shape[0]))
    distances, indices = nbrs.kneighbors(X)
    for i in range(X.shape[0]):
        for j in indices[i]:
            W[i, j] = 1
            W[j, i] = 1  # Symmetric connection

    # Degree matrix D (diagonal)
    D = np.diag(W.sum(axis=1))

    # Laplacian matrix L = D - W
    L = D - W
    return th.tensor(L, dtype=th.float64)


def compute_weighting_matrix(X, k=5):
    """Compute the weighting matrix W using k-nearest neighbors for data X."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    W = np.zeros((X.shape[0], X.shape[0]))
    distances, indices = nbrs.kneighbors(X)
    for i in range(X.shape[0]):
        for j in indices[i]:
            W[i, j] = 1
            W[j, i] = 1  # Symmetric connection
    return W  


class Accuracy:
    
    @staticmethod
    def knn_accuracy(train_data, train_labels, test_data, test_labels):
        num_classes = len(np.unique(train_labels))
        knn = KNeighborsClassifier(n_neighbors=num_classes)
        knn.fit(train_data, train_labels)
        predicted_labels = knn.predict(test_data)
        accuracy = accuracy_score(test_labels, predicted_labels)
        return accuracy, predicted_labels

    @staticmethod
    def roc_auc_f1(train_data, train_labels, test_data, test_labels):
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(train_data, train_labels)
        predicted_labels = knn.predict(test_data)
        roc_auc = roc_auc_score(test_labels, predicted_labels, multi_class='ovr')
        f1 = f1_score(test_labels, predicted_labels, average='weighted')
        return roc_auc, f1
    

class PLS(object):
    def __init__(self, n_components, solver, max_iter=500, tol=1e-06):
        self.n_components = n_components
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol

    def _fit(self, X, Y):
        X = X.clone()
        Y = Y.clone()
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]
        
        print('xxxxxxxxxxxxxxxyyyyyyyyyyyyy')
        print('FX',X.shape)
        print('Fy',Y.shape)
        print('N X ROW',n)
        print('P X COL',p)
        print('Q Y COL',q)

        n_components = self.n_components

        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        Xk, Yk, self._x_mean, self._y_mean = X, Y, x_mean, y_mean

        self.x_weights_ = th.zeros((p, n_components))
        self.y_weights_ = th.zeros((q, n_components))
        self._x_scores = th.zeros((n, n_components))
        self._y_scores = th.zeros((n, n_components))
        self.x_loadings_ = th.zeros((p, n_components))
        self.y_loadings_ = th.zeros((q, n_components))
        self.n_iter_ = []

        Y_eps = th.finfo(th.float64).eps
        for k in range(n_components):
            Yk_mask = th.all(th.abs(Yk) < 10 * Y_eps, dim=0)
            Yk[:, Yk_mask] = 0.0
            if self.solver == 'iter':
                x_weights, y_weights, n_iter_ = _get_first_singular_vectors_power_method(
                    Xk, Yk, max_iter=self.max_iter, tol=self.tol
                )
                self.n_iter_.append(n_iter_)
            elif self.solver == 'svd':
                x_weights, y_weights = _get_first_singular_vectors_svd(Xk, Yk)
            else:
                raise NameError('PLS solver not supported')

            _svd_flip_1d(x_weights, y_weights)
            x_scores = th.matmul(Xk, x_weights)
            y_ss = th.matmul(y_weights, y_weights)
            y_scores = th.matmul(Yk, y_weights) / y_ss
            x_loadings = th.matmul(x_scores, Xk) / th.matmul(x_scores, x_scores)
            Xk -= th.einsum('i,j->ij', x_scores, x_loadings)
            y_loadings = th.matmul(x_scores, Yk) / th.matmul(x_scores, x_scores)
            Yk -= th.einsum('i,j->ij', x_scores, y_loadings)

            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self._x_scores[:, k] = x_scores
            self._y_scores[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings
            self.x_scores_ = self._x_scores
            self.y_scores_ = self._y_scores

        self.x_rotations_ = th.matmul(
            self.x_weights_,
            th.pinverse(th.matmul(self.x_loadings_.T, self.x_weights_)))
        self.y_rotations_ = th.matmul(
            self.y_weights_, th.pinverse(th.matmul(self.y_loadings_.T, self.y_weights_)))

        self.coef_ = th.matmul(self.x_rotations_, self.y_loadings_.T)
        return self

    def fit(self, X, Y):
        self._fit(X, Y)
        return self

    def transform(self, X):
        X = X.clone()
        X -= self._x_mean
        x_scores = th.matmul(X, self.x_rotations_)
        return x_scores

class DeepPLS(object):
    def __init__(self, lv_dimensions, pls_solver, use_nonlinear_mapping, mapping_dimensions, nys_gamma_values, stack_previous_lv1):
        self.lv_dimensions = lv_dimensions
        self.n_layers = len(self.lv_dimensions)
        self.pls_solver = pls_solver
        self.latent_variables = []
        self.pls_funcs = []
        self.use_nonlinear_mapping = use_nonlinear_mapping
        self.mapping_dimensions = mapping_dimensions
        self.nys_gamma_values = nys_gamma_values
        self.mapping_funcs = []
        self.stack_previous_lv1 = stack_previous_lv1
        self.final_x_scores = None
        self.final_y_scores = None

        if self.use_nonlinear_mapping:
            assert len(self.lv_dimensions) == len(self.mapping_dimensions)
            assert len(self.mapping_dimensions) == len(self.nys_gamma_values)

    def _fit(self, X, Y):
        for layer_index in range(self.n_layers):
            if self.use_nonlinear_mapping:
                # Ensure n_components does not exceed the number of samples
                n_samples = X.shape[0]
                print("aaa",X.shape)
                print("bbb",Y.shape)
                n_components = min(self.mapping_dimensions[layer_index], n_samples)

                nys_func = Nystroem(kernel='rbf', gamma=self.nys_gamma_values[layer_index],
                                    n_components=n_components, n_jobs=-1)
                X_backup = X.clone()
                X = nys_func.fit_transform(X.numpy())  # Convert to numpy for sklearn
                self.mapping_funcs.append(nys_func)
                X = th.tensor(X)

                ########################################
                # Y = nys_func.fit_transform(Y.numpy())  # Convert to numpy for sklearn
                # self.mapping_funcs.append(nys_func)
                # Y = th.tensor(Y)
                ########################################
                print('a',X.shape)
                print('b',Y.shape)
                if self.stack_previous_lv1 and layer_index > 0:
                    lv1_previous_layer = X_backup[:, [0]]
                    X = th.hstack((lv1_previous_layer, X))     #normal
                    
                    # NEW  laplacian 1
                    # L_x = compute_laplacian(X_backup.numpy(), k=5)
                    # L_x = L_x.float()
                    # lambda_ = 1  # Regularization strength
                    # lv1_regularized = th.matmul(L_x, lv1_previous_layer)
                    # # lv1_regularized = lv1_previous_layer - lambda_ * th.matmul(L_x, lv1_previous_layer)
                    # X = th.hstack((lv1_regularized, X))


                    # # NEW  laplacian 1
                    # L_lv = compute_laplacian(lv1_previous_layer.numpy(), k=5)
                    # L_lv = L_lv.float()
                    # lambda_ = 1  # Regularization strength
                    # lv1_regularized = lv1_previous_layer - lambda_ * ((th.matmul(lv1_previous_layer.T, L_lv)).T)
                    # X = th.hstack((lv1_regularized, X))
                    
                     # NEW  laplacian 3
                    # L_lv = compute_laplacian(lv1_previous_layer.numpy(), k=5)
                    # L_lv = L_lv.float()
                    # lv1_regularized = th.matmul(L_lv, lv1_previous_layer)
                    # X = th.hstack((lv1_regularized, X))


                    #  # NEW  laplacian 2
                    # L_lv = compute_laplacian(lv1_previous_layer.numpy(), k=5)
                    # L_lv = L_lv.float()
                    # lambda_ = 0.6  # Regularization strength
                    # lv1_regularized = lv1_previous_layer - lambda_ * th.matmul(L_lv, lv1_previous_layer)
                    # X = th.hstack((lv1_regularized, X))




            # print('a2',X.shape)  
            # print('b2',Y.shape)
            pls = PLS(n_components=self.lv_dimensions[layer_index], solver=self.pls_solver)
            # print('a3',X.shape)
            # print('y', Y.shape)
            pls.fit(X, Y)
            
            self.pls_funcs.append(pls)

            latent_variables = pls.x_scores_
            self.latent_variables.append(latent_variables)
            X = latent_variables

        # Save final layer scores
        self.final_x_scores = self.latent_variables[-1]
        self.final_y_scores = pls.y_scores_

    def fit(self, X, Y):
        self._fit(X, Y)
        return self

    def get_final_scores(self):
        return self.final_x_scores, self.final_y_scores