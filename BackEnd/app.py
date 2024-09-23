# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import torch as th
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from models import DeepPLS, PLS

# app = Flask(__name__)
# CORS(app)


# def knn_accuracy(train_data, train_labels, test_data, test_labels):
#     num_classes = len(np.unique(train_labels))
#     knn = KNeighborsClassifier(n_neighbors=1)
#     knn.fit(train_data, train_labels)
#     predicted_labels = knn.predict(test_data)
#     accuracy = accuracy_score(test_labels, predicted_labels)
#     return accuracy

# def roc_auc_f1(train_data, train_labels, test_data, test_labels):
#     num_classes = len(np.unique(train_labels))
#     knn = KNeighborsClassifier(n_neighbors=1)
#     knn.fit(train_data, train_labels)
#     predicted_labels = knn.predict(test_data)
#     roc_auc = roc_auc_score(test_labels, predicted_labels, multi_class='ovr')
#     f1 = f1_score(test_labels, predicted_labels, average='weighted')
#     return roc_auc, f1


# @app.route('/upload', methods=['POST'])
# def upload_files():
#     if 'source' not in request.files or 'target' not in request.files:
#         return jsonify({'error': 'Source and target files are required'}), 400
    
#     source_file = request.files['source']
#     target_file = request.files['target']

#     source_data_df = pd.read_excel(source_file)
#     target_data_df = pd.read_excel(target_file)

#     # Convert data to tensors and process
#     source_data = th.tensor(source_data_df.iloc[:, :-1].values, dtype=th.float32)
#     source_label = th.tensor(source_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

#     target_data = th.tensor(target_data_df.iloc[:, :-1].values, dtype=th.float32)
#     target_label = th.tensor(target_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

#     # Check for NaNs or Infs and replace them
#     source_data[th.isnan(source_data)] = 0
#     source_data[th.isinf(source_data)] = 0
#     target_data[th.isnan(target_data)] = 0
#     target_data[th.isinf(target_data)] = 0

#     # Standardize the data
#     source_data = (source_data - source_data.mean(dim=0)) / source_data.std(dim=0)
#     target_data = (target_data - target_data.mean(dim=0)) / target_data.std(dim=0)

#     # Fix random seed
#     th.manual_seed(0)

#     # Add small noise to handle ill-conditioning
#     noise_level = 1e-5
#     source_data += noise_level * th.randn(source_data.size())
#     target_data += noise_level * th.randn(target_data.size())

#     # TRANSPOSE
#     source_data_transpose = source_data.T
#     target_data_transpose = target_data.T

#     #PLS
#     ##################PLS START
#     # Separate PLS fitting for final scores
#     standalone_pls = PLS(n_components=30, solver='iter')  # Adjust n_components as needed
#     standalone_pls.fit(source_data_transpose, target_data_transpose)

#     # Retrieve final scores from standalone PLS
#     standalone_pls_x_scores = standalone_pls.x_scores_
#     standalone_pls_y_scores = standalone_pls.y_scores_
#     # print('Standalone PLS x_scores:', standalone_pls_x_scores)
#     # print('Standalone PLS y_scores:', standalone_pls_y_scores)

#     plsSource=th.matmul(source_data,standalone_pls_x_scores)
#     plsTarget=th.matmul(target_data,standalone_pls_y_scores)
#     #PLs tensor
#     plsSource_np=plsSource.numpy()
#     plsTarget_np=plsTarget.numpy()
#     Pls_source_label_np = source_label.numpy().ravel()
#     Pls_target_label_np = target_label.numpy().ravel()
#     # Calculate metrics using KNN classifier
#     pls_accuracy = knn_accuracy(plsSource_np, Pls_source_label_np, plsTarget_np, Pls_target_label_np)
#     pls_auc_roc, pls_f1_score_macro = roc_auc_f1(plsSource_np, Pls_source_label_np, plsTarget_np, Pls_target_label_np)
#     # PLS ends


#     # DPLS
#     Deep_pls = DeepPLS(
#         lv_dimensions=[30, 40],
#         pls_solver='svd',
#         use_nonlinear_mapping=False,
#         mapping_dimensions=[800, 800],
#         nys_gamma_values=[0.014, 2.8],
#         stack_previous_lv1=True
#     )

#     Deep_pls.fit(source_data_transpose, target_data_transpose)
#     # Retrieve final layer scores
#     dpls_final_x_scores, dpls_final_y_scores = Deep_pls.get_final_scores()
#     # MATRIX MULTIPLICATION
#     dpls_Final_source_data = th.matmul(source_data, dpls_final_x_scores)
#     dpls_Final_target_data = th.matmul(target_data, dpls_final_y_scores)
#     # Convert tensors to numpy arrays for sklearn
#     dpls_Final_source_data_np = dpls_Final_source_data.numpy()
#     dpls_Final_target_data_np = dpls_Final_target_data.numpy()
#     dpls_source_label_np = source_label.numpy().ravel()
#     dpls_target_label_np = target_label.numpy().ravel()
#     # Calculate metrics using KNN classifier
#     dpls_accuracy = knn_accuracy(dpls_Final_source_data_np, dpls_source_label_np, dpls_Final_target_data_np, dpls_target_label_np)
#     dpls_auc_roc, dpls_f1_score_macro = roc_auc_f1(dpls_Final_source_data_np, dpls_source_label_np, dpls_Final_target_data_np, dpls_target_label_np)
#     # DPLS ends


#     # GDPLS start
#     generalized_deep_pls = DeepPLS(
#         lv_dimensions=[30, 40],
#         pls_solver='svd',
#         use_nonlinear_mapping=True,
#         mapping_dimensions=[800, 800],
#         nys_gamma_values=[0.014, 2.8],
#         stack_previous_lv1=True
#     )

#     generalized_deep_pls.fit(source_data_transpose, target_data_transpose)

#     # Retrieve final layer scores
#     final_x_scores, final_y_scores = generalized_deep_pls.get_final_scores()

#     # MATRIX MULTIPLICATION
#     Final_source_data = th.matmul(source_data, final_x_scores)
#     Final_target_data = th.matmul(target_data, final_y_scores)

#     # Convert tensors to numpy arrays for sklearn
#     Final_source_data_np = Final_source_data.numpy()
#     Final_target_data_np = Final_target_data.numpy()
#     source_label_np = source_label.numpy().ravel()
#     target_label_np = target_label.numpy().ravel()


#     # Calculate metrics using KNN classifier
#     accuracy = knn_accuracy(Final_source_data_np, source_label_np, Final_target_data_np, target_label_np)
#     auc_roc, f1_score_macro = roc_auc_f1(Final_source_data_np, source_label_np, Final_target_data_np, target_label_np)
    
#     # GDPLS ends

#     # Return all three metrics in the response
#     return jsonify({
#         'accuracy': accuracy,
#         'aucRoc': auc_roc,
#         'f1Score': f1_score_macro,
#         'pls_accuracy': pls_accuracy,
#         'pls_auc_roc': pls_auc_roc,
#         'pls_f1_score': pls_f1_score_macro,
#         'dpls_accuracy': dpls_accuracy,
#         'dpls_auc_roc': dpls_auc_roc,
#         'dpls_f1_score': dpls_f1_score_macro
#     })
#     # return jsonify({
#     #   'accuracy': accuracy,
#     #   'aucRoc': auc_roc,
#     #   'f1Score': f1_score_macro
#     #  })



#     # return jsonify({'accuracy': accuracy})

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import torch as th
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from models import PLS, DeepPLS

# app = Flask(__name__)
# CORS(app)

# def knn_accuracy(train_data, train_labels, test_data, test_labels):
#     knn = KNeighborsClassifier(n_neighbors=1)
#     knn.fit(train_data, train_labels)
#     predicted_labels = knn.predict(test_data)
#     accuracy = accuracy_score(test_labels, predicted_labels)
#     return accuracy

# def roc_auc_f1(train_data, train_labels, test_data, test_labels):
#     knn = KNeighborsClassifier(n_neighbors=1)
#     knn.fit(train_data, train_labels)
#     predicted_labels = knn.predict(test_data)
#     roc_auc = roc_auc_score(test_labels, predicted_labels, multi_class='ovr')
#     f1 = f1_score(test_labels, predicted_labels, average='weighted')
#     return roc_auc, f1

# def process_files(request):
#     if 'source' not in request.files or 'target' not in request.files:
#         return None, None, None, None, 'Source and target files are required'

#     source_file = request.files['source']
#     target_file = request.files['target']

#     source_data_df = pd.read_excel(source_file)
#     target_data_df = pd.read_excel(target_file)

#     source_data = th.tensor(source_data_df.iloc[:, :-1].values, dtype=th.float32)
#     source_label = th.tensor(source_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

#     target_data = th.tensor(target_data_df.iloc[:, :-1].values, dtype=th.float32)
#     target_label = th.tensor(target_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

#     source_data[th.isnan(source_data)] = 0
#     source_data[th.isinf(source_data)] = 0
#     target_data[th.isnan(target_data)] = 0
#     target_data[th.isinf(target_data)] = 0

#     # source_data = (source_data - source_data.mean(dim=0)) / source_data.std(dim=0)
#     # target_data = (target_data - target_data.mean(dim=0)) / target_data.std(dim=0)

#     return source_data, source_label, target_data, target_label, None

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     model_type = request.form.get('model_type')
#     if model_type not in ['pls', 'dpls', 'gdpls']:
#         return jsonify({'error': 'Invalid model type'}), 400

#     source_data, source_label, target_data, target_label, error = process_files(request)
#     if error:
#         return jsonify({'error': error}), 400

#     source_data_transpose = source_data.T
#     target_data_transpose = target_data.T

#     if model_type == 'pls':
#         model = PLS(n_components=30, solver='svd')
#         model.fit(source_data_transpose, target_data_transpose)
#         final_x_scores = model.x_scores_
#         final_y_scores = model.y_scores_
#     elif model_type == 'dpls':
#         model = DeepPLS(
#             lv_dimensions=[30, 40],
#             pls_solver='svd',
#             use_nonlinear_mapping=False,
#             mapping_dimensions=[800, 800],
#             nys_gamma_values=[0.014, 2.8],
#             stack_previous_lv1=True
#         )
#         model.fit(source_data_transpose, target_data_transpose)
#         final_x_scores, final_y_scores = model.get_final_scores()
#     elif model_type == 'gdpls':
#         model = DeepPLS(
#             lv_dimensions=[30, 40],
#             pls_solver='svd',
#             use_nonlinear_mapping=True,
#             mapping_dimensions=[800, 800],
#             nys_gamma_values=[0.014, 2.8],
#             stack_previous_lv1=True
#         )
#         model.fit(source_data_transpose, target_data_transpose)
#         final_x_scores, final_y_scores = model.get_final_scores()

#     transformed_source_data = th.matmul(source_data, final_x_scores)
#     transformed_target_data = th.matmul(target_data, final_y_scores)

#     transformed_source_data_np = transformed_source_data.numpy()
#     transformed_target_data_np = transformed_target_data.numpy()
#     source_label_np = source_label.numpy().ravel()
#     target_label_np = target_label.numpy().ravel()

#     accuracy = knn_accuracy(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
#     auc_roc, f1_score_macro = roc_auc_f1(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)

#     return jsonify({
#         'accuracy': accuracy,
#         'aucRoc': auc_roc,
#         'f1Score': f1_score_macro
#     })

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch as th
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from models import PLS, DeepPLS
from coral import CORAL
from tca import TCA
from jda import JDA

app = Flask(__name__)
CORS(app)

def knn_accuracy(train_data, train_labels, test_data, test_labels):
    num_classes = len(np.unique(train_labels))
    knn = KNeighborsClassifier(n_neighbors=num_classes)
    knn.fit(train_data, train_labels)
    predicted_labels = knn.predict(test_data)
    accuracy = accuracy_score(test_labels, predicted_labels)
    return accuracy, predicted_labels

def roc_auc_f1(train_data, train_labels, test_data, test_labels):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_data, train_labels)
    predicted_labels = knn.predict(test_data)
    roc_auc = roc_auc_score(test_labels, predicted_labels, multi_class='ovr')
    f1 = f1_score(test_labels, predicted_labels, average='weighted')
    return roc_auc, f1

def process_files(request):
    if 'source' not in request.files or 'target' not in request.files:
        return None, None, None, None, 'Source and target files are required'

    source_file = request.files['source']
    target_file = request.files['target']

    source_data_df = pd.read_excel(source_file)
    target_data_df = pd.read_excel(target_file)

    source_data = th.tensor(source_data_df.iloc[:, :-1].values, dtype=th.float32)
    source_label = th.tensor(source_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

    target_data = th.tensor(target_data_df.iloc[:, :-1].values, dtype=th.float32)
    target_label = th.tensor(target_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

    source_data[th.isnan(source_data)] = 0
    source_data[th.isinf(source_data)] = 0
    target_data[th.isnan(target_data)] = 0
    target_data[th.isinf(target_data)] = 0

    source_data = (source_data - source_data.mean(dim=0)) / source_data.std(dim=0)
    target_data = (target_data - target_data.mean(dim=0)) / target_data.std(dim=0)

    return source_data, source_label, target_data, target_label, target_data_df, None

@app.route('/upload', methods=['POST'])
def upload_files():
    model_type = request.form.get('model_type')
    if model_type not in ['pls', 'dpls', 'gdpls', 'coral','tca','jda']:
        return jsonify({'error': 'Invalid model type'}), 400

    source_data, source_label, target_data, target_label, target_data_df, error = process_files(request)
    if error:
        return jsonify({'error': error}), 400

    source_data_transpose = source_data.T
    target_data_transpose = target_data.T

    if model_type == 'pls':
        model = PLS(n_components=30, solver='iter')
        model.fit(source_data_transpose, target_data_transpose)
        final_x_scores = model.x_scores_
        final_y_scores = model.y_scores_
        transformed_source_data = th.matmul(source_data, final_x_scores)
        transformed_target_data = th.matmul(target_data, final_y_scores)

        transformed_source_data_np = transformed_source_data.numpy()
        transformed_target_data_np = transformed_target_data.numpy()
        source_label_np = source_label.numpy().ravel()
        target_label_np = target_label.numpy().ravel()

        accuracy, predicted_labels = knn_accuracy(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        auc_roc, f1_score_macro = roc_auc_f1(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        target_data_with_labels = target_data_df.copy()
        target_data_with_labels['Predicted Label'] = predicted_labels
        target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == predicted_labels, 'Correct', 'Incorrect')

        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')




    elif model_type == 'dpls':
        model = DeepPLS(
            lv_dimensions=[30, 40],
            pls_solver='iter',
            use_nonlinear_mapping=False,
            mapping_dimensions=[800, 800],
            nys_gamma_values=[0.014, 2.8],
            stack_previous_lv1=True
        )
        
        model.fit(source_data_transpose, target_data_transpose)
        final_x_scores, final_y_scores = model.get_final_scores()
        transformed_source_data = th.matmul(source_data, final_x_scores)
        transformed_target_data = th.matmul(target_data, final_y_scores)

        transformed_source_data_np = transformed_source_data.numpy()
        transformed_target_data_np = transformed_target_data.numpy()
        source_label_np = source_label.numpy().ravel()
        target_label_np = target_label.numpy().ravel()

        accuracy, predicted_labels = knn_accuracy(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        auc_roc, f1_score_macro = roc_auc_f1(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        target_data_with_labels = target_data_df.copy()
        target_data_with_labels['Predicted Label'] = predicted_labels
        target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == predicted_labels, 'Correct', 'Incorrect')

        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')


    elif model_type == 'gdpls':
        model = DeepPLS(
            lv_dimensions=[30, 40],
            pls_solver='iter',
            use_nonlinear_mapping=True,
            mapping_dimensions=[800, 800],
            nys_gamma_values=[0.014, 2.8],
            stack_previous_lv1=True
        )
        model.fit(source_data_transpose, target_data_transpose)
        final_x_scores, final_y_scores = model.get_final_scores()
        transformed_source_data = th.matmul(source_data, final_x_scores)
        transformed_target_data = th.matmul(target_data, final_y_scores)

        transformed_source_data_np = transformed_source_data.numpy()
        transformed_target_data_np = transformed_target_data.numpy()
        source_label_np = source_label.numpy().ravel()
        target_label_np = target_label.numpy().ravel()

        accuracy, predicted_labels = knn_accuracy(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        auc_roc, f1_score_macro = roc_auc_f1(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        target_data_with_labels = target_data_df.copy()
        target_data_with_labels['Predicted Label'] = predicted_labels
        target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == predicted_labels, 'Correct', 'Incorrect')
        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')

    elif model_type == 'coral':
        print('aaa')
        source_file = request.files['source']
        target_file = request.files['target']
        source_data_df1 = pd.read_excel(source_file)
        target_data_df1 = pd.read_excel(target_file)
        # Split features and labels
        Xs = source_data_df1.iloc[:, :-1].values  # Source features
        Ys = source_data_df1.iloc[:, -1].values   # Source labels
        Xt = target_data_df1.iloc[:, :-1].values  # Target features
        Yt = target_data_df1.iloc[:, -1].values   # Target labels
        model = CORAL()
        accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(Xs, Ys, Xt, Yt)
        target_data_with_labels = target_data_df1.copy()
        target_data_with_labels['Predicted Label'] = ypre
        target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')



    elif model_type == 'tca':
        print('a1a1a1')
        source_file = request.files['source']
        target_file = request.files['target']
        source_data_df1 = pd.read_excel(source_file)
        target_data_df1 = pd.read_excel(target_file)
        # Split features and labels
        Xs = source_data_df1.iloc[:, :-1].values  # Source features
        Ys = source_data_df1.iloc[:, -1].values   # Source labels
        Xt = target_data_df1.iloc[:, :-1].values  # Target features
        Yt = target_data_df1.iloc[:, -1].values   # Target labels
        model = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
        accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(Xs, Ys, Xt, Yt)
        target_data_with_labels = target_data_df1.copy()
        target_data_with_labels['Predicted Label'] = ypre
        target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')



    elif model_type == 'jda':
        print('jdaaaa')
        source_file = request.files['source']
        target_file = request.files['target']
        source_data_df1 = pd.read_excel(source_file)
        target_data_df1 = pd.read_excel(target_file)
        # Split features and labels
        Xs = source_data_df1.iloc[:, :-1].values  # Source features
        Ys = source_data_df1.iloc[:, -1].values   # Source labels
        Xt = target_data_df1.iloc[:, :-1].values  # Target features
        Yt = target_data_df1.iloc[:, -1].values   # Target labels
        model = JDA(kernel_type='primal', dim=30, lamb=1, gamma=1)
        accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(Xs, Ys, Xt, Yt)
        target_data_with_labels = target_data_df1.copy()
        target_data_with_labels['Predicted Label'] = ypre
        target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')





    # transformed_source_data = th.matmul(source_data, final_x_scores)
    # transformed_target_data = th.matmul(target_data, final_y_scores)

    # transformed_source_data_np = transformed_source_data.numpy()
    # transformed_target_data_np = transformed_target_data.numpy()
    # source_label_np = source_label.numpy().ravel()
    # target_label_np = target_label.numpy().ravel()

    # accuracy, predicted_labels = knn_accuracy(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
    # auc_roc, f1_score_macro = roc_auc_f1(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
    # target_data_with_labels = target_data_df.copy()
    # target_data_with_labels['Predicted Label'] = predicted_labels
    # target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == predicted_labels, 'Correct', 'Incorrect')

    # target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')

    return jsonify({
        'accuracy': accuracy,
        'aucRoc': auc_roc,
        'f1Score': f1_score_macro,
        'targetDataWithLabels': target_data_with_labels_dict
    })

if __name__ == '__main__':
    app.run(debug=True)
