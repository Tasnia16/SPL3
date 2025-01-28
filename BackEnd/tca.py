
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        auc_roc = roc_auc_score(Yt, y_pred, multi_class='ovr')
        f1 = f1_score(Yt, y_pred, average='weighted')
        # f1 = f1_score(Yt, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        return acc, y_pred , auc_roc , f1

    #222222222222222222222222222222

    # def fit_predict(self, Xs, Ys, Xt, Yt):
    #     Xs_new, Xt_new = self.fit(Xs, Xt)
    #     clf = KNeighborsClassifier(n_neighbors=1)
    #     clf.fit(Xs_new, Ys.ravel())
    #     y_pred = clf.predict(Xt_new)
    #     acc = accuracy_score(Yt, y_pred)
        
    #     # Handle mismatched labels for ROC AUC and F1
    #     unique_labels = sorted(set(Ys.ravel()).union(set(Yt.ravel())))  # Combine all unique labels
    #     try:
    #         # Ensure compatibility for ROC AUC
    #         y_true_one_hot = np.zeros((len(Yt), len(unique_labels)))
    #         y_pred_one_hot = np.zeros((len(y_pred), len(unique_labels)))
    #         label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            
    #         for i, label in enumerate(Yt.ravel()):
    #             y_true_one_hot[i, label_to_index[label]] = 1
    #         for i, label in enumerate(y_pred.ravel()):
    #             y_pred_one_hot[i, label_to_index[label]] = 1

    #         auc_roc = roc_auc_score(y_true_one_hot, y_pred_one_hot, multi_class='ovr')
    #     except ValueError as e:
    #         # Handle error when ROC AUC cannot be calculated
    #         print(f"Warning: ROC AUC computation failed due to {e}")
    #         auc_roc = None

    #     # Compute weighted F1 score
    #     f1 = f1_score(Yt, y_pred, average='weighted', zero_division=0)

    #     return acc, y_pred, auc_roc, f1

##33333333333333333333333333333333333

    # def fit_predict(self, Xs, Ys, Xt, Yt, unknown_threshold=0.5):
    #     """
    #     Fits the model, predicts labels for the target, and handles unknown labels.

    #     Args:
    #         Xs: Source data features.
    #         Ys: Source data labels.
    #         Xt: Target data features.
    #         Yt: Target data labels (for evaluation).
    #         unknown_threshold: Threshold for assigning "unknown" labels based on confidence or distance.

    #     Returns:
    #         acc: Accuracy of known label predictions.
    #         y_pred_with_unknown: Predictions with unknown labels assigned.
    #         auc_roc: ROC AUC score for known labels.
    #         f1: Weighted F1 score.
    #     """
    #     # Fit and transform the data
    #     Xs_new, Xt_new = self.fit(Xs, Xt)
    #     clf = KNeighborsClassifier(n_neighbors=1)
    #     clf.fit(Xs_new, Ys.ravel())

    #     # Predict labels and distances
    #     distances, indices = clf.kneighbors(Xt_new, n_neighbors=1)
    #     y_pred = clf.predict(Xt_new)

    #     # Initialize "unknown" label handling
    #     y_pred_with_unknown = []
    #     for dist, pred in zip(distances.ravel(), y_pred):
    #         if dist > unknown_threshold:  # Assign "unknown" if the distance is high
    #             y_pred_with_unknown.append(-1)  # Use -1 for "unknown" labels
    #         else:
    #             y_pred_with_unknown.append(pred)

    #     # Compute metrics
    #     # Accuracy: Ignore "unknown" labels in evaluation
    #     valid_indices = [i for i, label in enumerate(Yt) if y_pred_with_unknown[i] != -1]
    #     y_true_valid = np.array(Yt)[valid_indices]
    #     y_pred_valid = np.array(y_pred_with_unknown)[valid_indices]

    #     acc = accuracy_score(y_true_valid, y_pred_valid) if len(valid_indices) > 0 else 0

    #     # Handle mismatched labels for ROC AUC and F1
    #     unique_labels = sorted(set(Ys.ravel()).union(set(Yt.ravel())))  # Combine all unique labels
    #     try:
    #         # Ensure compatibility for ROC AUC
    #         y_true_one_hot = np.zeros((len(Yt), len(unique_labels) + 1))  # +1 for "unknown" label
    #         y_pred_one_hot = np.zeros((len(y_pred_with_unknown), len(unique_labels) + 1))
    #         label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    #         label_to_index[-1] = len(unique_labels)  # Assign index to "unknown"

    #         for i, label in enumerate(Yt.ravel()):
    #             y_true_one_hot[i, label_to_index[label]] = 1
    #         for i, label in enumerate(y_pred_with_unknown):
    #             y_pred_one_hot[i, label_to_index[label]] = 1

    #         auc_roc = roc_auc_score(y_true_one_hot, y_pred_one_hot, multi_class='ovr')
    #     except ValueError as e:
    #         # Handle error when ROC AUC cannot be calculated
    #         print(f"Warning: ROC AUC computation failed due to {e}")
    #         auc_roc = None

    #     # Compute weighted F1 score
    #     f1 = f1_score(
    #         y_true_valid,
    #         y_pred_valid,
    #         labels=[label for label in unique_labels if label in y_true_valid],
    #         average='weighted',
    #         zero_division=0,
    #     )

    #     return acc, y_pred_with_unknown, auc_roc, f1
