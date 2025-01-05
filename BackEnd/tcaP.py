# tcap.py
import numpy as np
from scipy.spatial.distance import pdist, squareform

class TCA_PLUS:
    
    @staticmethod
    def normalization_None(X):
        """ No Normalization applied """
        return X

    @staticmethod
    def normalization_N1(X):
        """ N1 normalization: min-max normalization """
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val)

    @staticmethod
    def normalization_N2(X):
        """ N2 normalization: standard score normalization """
        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        return (X - mean) / (std_dev)

    @staticmethod
    def normalization_N3(X, X_src):
        """ N3 normalization: normalize based on source project statistics """
        mean_src = np.mean(X_src, axis=0)
        std_src = np.std(X_src, axis=0)
        return (X - mean_src) / (std_src)

    @staticmethod
    def normalization_N4(X, X_tar):
        """ N4 normalization: normalize based on target project statistics """
        mean_tar = np.mean(X_tar, axis=0)
        std_tar = np.std(X_tar, axis=0)
        return (X - mean_tar) / (std_tar)

    @staticmethod
    def apply_normalization(X_src, X_tar, similarity_vector):
        """ Apply the normalization based on the similarity vector and rules """
        dist_mean, dist_median, dist_min, dist_max, dist_std, num_instances = similarity_vector
        
        if dist_mean == "SAME" and dist_std == "SAME":
            return TCA_PLUS.normalization_None(X_src), TCA_PLUS.normalization_None(X_tar)
        
        if num_instances in ["MUCH MORE", "MUCH LESS"] and dist_min in ["MUCH MORE", "MUCH LESS"] and dist_max in ["MUCH MORE", "MUCH LESS"]:
            return TCA_PLUS.normalization_N1(X_src), TCA_PLUS.normalization_N1(X_tar)
        
        if (dist_std == "MUCH MORE" and num_instances == "LESS") or (dist_std == "MUCH LESS" and num_instances == "MORE"):
            return TCA_PLUS.normalization_N3(X_src, X_src), TCA_PLUS.normalization_N3(X_tar, X_src)
        
        if (dist_std == "MUCH MORE" and num_instances == "MUCH MORE") or (dist_std == "MUCH LESS" and num_instances == "MUCH LESS"):
            return TCA_PLUS.normalization_N4(X_src, X_tar), TCA_PLUS.normalization_N4(X_tar, X_tar)
        
        return TCA_PLUS.normalization_N2(X_src), TCA_PLUS.normalization_N2(X_tar)

    @staticmethod
    def compute_characteristic_vector(X):
        """ Compute characteristic vector for the dataset """
        distances = pdist(X, metric='euclidean')
        dist_matrix = squareform(distances)

        dist_mean = np.mean(distances)
        dist_median = np.median(distances)
        dist_min = np.min(distances)
        dist_max = np.max(distances)
        dist_std = np.std(distances)
        num_instances = X.shape[0]

        return {
            'dist_mean': dist_mean,
            'dist_median': dist_median,
            'dist_min': dist_min,
            'dist_max': dist_max,
            'dist_std': dist_std,
            'num_instances': num_instances
        }

    @staticmethod
    def assign_nominal_values(cS, cT):
        """ Assign nominal values to the similarity vector """
        similarity_vector = {}

        for key in cS:
            if cS[key] * 1.6 < cT[key]:
                similarity_vector[key] = "MUCH MORE"
            elif cS[key] * 1.3 < cT[key] <= cS[key] * 1.6:
                similarity_vector[key] = "MORE"
            elif cS[key] * 1.1 < cT[key] <= cS[key] * 1.3:
                similarity_vector[key] = "SLIGHTLY MORE"
            elif cS[key] * 0.9 <= cT[key] <= cS[key] * 1.1:
                similarity_vector[key] = "SAME"
            elif cS[key] * 0.7 <= cT[key] < cS[key] * 0.9:
                similarity_vector[key] = "SLIGHTLY LESS"
            elif cS[key] * 0.4 <= cT[key] < cS[key] * 0.7:
                similarity_vector[key] = "LESS"
            else:
                similarity_vector[key] = "MUCH LESS"

        return similarity_vector
