from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch as th
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from models import DeepPLS

app = Flask(__name__)
CORS(app)


def knn_accuracy(train_data, train_labels, test_data, test_labels):
    num_classes = len(np.unique(train_labels))
    knn = KNeighborsClassifier(n_neighbors=num_classes)
    knn.fit(train_data, train_labels)
    predicted_labels = knn.predict(test_data)
    accuracy = accuracy_score(test_labels, predicted_labels)
    return accuracy

def roc_auc_f1(train_data, train_labels, test_data, test_labels):
    num_classes = len(np.unique(train_labels))
    knn = KNeighborsClassifier(n_neighbors=num_classes)
    knn.fit(train_data, train_labels)
    predicted_labels = knn.predict(test_data)
    roc_auc = roc_auc_score(test_labels, predicted_labels, multi_class='ovr')
    f1 = f1_score(test_labels, predicted_labels, average='weighted')
    return roc_auc, f1


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'source' not in request.files or 'target' not in request.files:
        return jsonify({'error': 'Source and target files are required'}), 400
    
    source_file = request.files['source']
    target_file = request.files['target']

    source_data_df = pd.read_excel(source_file)
    target_data_df = pd.read_excel(target_file)

    # Convert data to tensors and process
    source_data = th.tensor(source_data_df.iloc[:, :-1].values, dtype=th.float32)
    source_label = th.tensor(source_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

    target_data = th.tensor(target_data_df.iloc[:, :-1].values, dtype=th.float32)
    target_label = th.tensor(target_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

    # Check for NaNs or Infs and replace them
    source_data[th.isnan(source_data)] = 0
    source_data[th.isinf(source_data)] = 0
    target_data[th.isnan(target_data)] = 0
    target_data[th.isinf(target_data)] = 0

    # Standardize the data
    source_data = (source_data - source_data.mean(dim=0)) / source_data.std(dim=0)
    target_data = (target_data - target_data.mean(dim=0)) / target_data.std(dim=0)

    # Fix random seed
    th.manual_seed(0)

    # Add small noise to handle ill-conditioning
    noise_level = 1e-5
    source_data += noise_level * th.randn(source_data.size())
    target_data += noise_level * th.randn(target_data.size())

    # TRANSPOSE
    source_data_transpose = source_data.T
    target_data_transpose = target_data.T

    # Define and fit the Deep PLS model
    generalized_deep_pls = DeepPLS(
        lv_dimensions=[30, 40],
        pls_solver='svd',
        use_nonlinear_mapping=True,
        mapping_dimensions=[800, 800],
        nys_gamma_values=[0.014, 2.8],
        stack_previous_lv1=True
    )

    generalized_deep_pls.fit(source_data_transpose, target_data_transpose)

    # Retrieve final layer scores
    final_x_scores, final_y_scores = generalized_deep_pls.get_final_scores()

    # MATRIX MULTIPLICATION
    Final_source_data = th.matmul(source_data, final_x_scores)
    Final_target_data = th.matmul(target_data, final_y_scores)

    # Convert tensors to numpy arrays for sklearn
    Final_source_data_np = Final_source_data.numpy()
    Final_target_data_np = Final_target_data.numpy()
    source_label_np = source_label.numpy().ravel()
    target_label_np = target_label.numpy().ravel()


    # Calculate metrics using KNN classifier
    accuracy = knn_accuracy(Final_source_data_np, source_label_np, Final_target_data_np, target_label_np)
    auc_roc, f1_score_macro = roc_auc_f1(Final_source_data_np, source_label_np, Final_target_data_np, target_label_np)


    # Return all three metrics in the response
    return jsonify({
      'accuracy': accuracy,
      'aucRoc': auc_roc,
      'f1Score': f1_score_macro
     })


    # return jsonify({'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)
