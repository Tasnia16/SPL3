
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import sklearn
# import torch as th
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# from models import PLS, DeepPLS,Accuracy
# from coral import CORAL
# from tca import TCA
# from jda import JDA
# from tcaP import TCA_PLUS

# app = Flask(__name__)
# CORS(app)

# def separate_features_and_labels(df):
#     """
#     Function to separate features and labels, assuming labels are in the first or last column.
#     """
#     # Check if label is in the first or last column
#     if df.columns[0].lower() in ['label', 'labels', 'classes','defect','defective','isdefective', 'class']:
#         label = df.iloc[:, 0].values
#         data = df.iloc[:, 1:].values
#     elif df.columns[-1].lower() in ['label', 'labels', 'classes','defect','defective','isdefective', 'class']:
#         label = df.iloc[:, -1].values
#         data = df.iloc[:, :-1].values
#     else:
#         return None, None, 'Could not detect label column (expecting it in the first or last column)'
    
#     return data, label, None


# def read_file(file):
#     """Helper function to read either Excel or CSV files."""
#     try:
#         if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
#             return pd.read_excel(file)
#         elif file.filename.endswith('.csv'):
#             print('aaaaaaaaaaaaaaaaaaaaaaa')
#             return pd.read_csv(file)
#         else:
#             raise ValueError("File must be an Excel or CSV format")
#     except Exception as e:
#         print(f"Error while reading the file: {e}")
#         raise



# def process_files(request):
#     if 'source' not in request.files or 'target' not in request.files:
#         return None, None, None, None, None, 'Source and target files are required'

#     source_file = request.files['source']
#     target_file = request.files['target']

#     # source_data_df = pd.read_excel(source_file)
#     # target_data_df = pd.read_excel(target_file)
#     try:
#         source_data_df = read_file(source_file)
#         target_data_df = read_file(target_file)
#     except ValueError as e:
#         return None, None, None, None, None, str(e)
    


#     # Separate features and labels for source and target data
#     source_data, source_label, error = separate_features_and_labels(source_data_df)
#     if error:
#         return None,None, None, None, None, error

#     target_data, target_label, error = separate_features_and_labels(target_data_df)
#     if error:
#         return None, None, None, None, None, error

#     # Convert to tensors for PyTorch compatibility
#     source_data = th.tensor(source_data, dtype=th.float32)
#     source_label = th.tensor(source_label[:, None], dtype=th.float32)
#     target_data = th.tensor(target_data, dtype=th.float32)
#     target_label = th.tensor(target_label[:, None], dtype=th.float32)
   
        


#     # source_data = th.tensor(source_data_df.iloc[:, :-1].values, dtype=th.float32)
#     # source_label = th.tensor(source_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

#     # target_data = th.tensor(target_data_df.iloc[:, :-1].values, dtype=th.float32)
#     # target_label = th.tensor(target_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

#     source_data[th.isnan(source_data)] = 0
#     source_data[th.isinf(source_data)] = 0
#     target_data[th.isnan(target_data)] = 0
#     target_data[th.isinf(target_data)] = 0

#     source_data = (source_data - source_data.mean(dim=0)) / source_data.std(dim=0)
#     target_data = (target_data - target_data.mean(dim=0)) / target_data.std(dim=0)

#     return source_data, source_label, target_data, target_label, target_data_df, None




# def process_files1(request):
#     if 'source' not in request.files or 'target' not in request.files:
#         return None, None, None, None, None, 'Source and target files are required'

#     source_file = request.files['source']
#     target_file = request.files['target']

#     try:
#         source_data_df = read_file(source_file)
#         target_data_df = read_file(target_file)
#     except ValueError as e:
#         return None, None, None, None, None, str(e)
    


#     # Separate features and labels for source and target data
#     source_data, source_label, error = separate_features_and_labels(source_data_df)
#     if error:
#         return None,None, None, None, None, error

#     target_data, target_label, error = separate_features_and_labels(target_data_df)
#     if error:
#         return None, None, None, None, None, error


#     # source_data_df = pd.read_excel(source_file)
#     # target_data_df = pd.read_excel(target_file)
  
#     # source_data = source_data_df.iloc[:, :-1].values
#     # source_label = source_data_df.iloc[:, -1].values
#     # target_data = target_data_df.iloc[:, :-1].values
#     # target_label = target_data_df.iloc[:, -1].values

#     return source_data, source_label, target_data, target_label, target_data_df, None    

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     model_type = request.form.get('model_type')
#     if model_type not in ['pls', 'dpls', 'gdpls', 'coral','tca','jda','tcaPlus']:
#         return jsonify({'error': 'Invalid model type'}), 400

#     source_data, source_label, target_data, target_label, target_data_df, error = process_files(request)
#     if error:
#         return jsonify({'error': error}), 400


    
#     source_data1, source_label1, target_data1, target_label1, target_data_df1, error = process_files1(request)
#     if error:
#         return jsonify({'error': error}), 400




#     source_data_transpose = source_data.T
#     target_data_transpose = target_data.T

#     if model_type == 'pls':
#         model = PLS(n_components=30, solver='iter')
#         model.fit(source_data_transpose, target_data_transpose)
#         final_x_scores = model.x_scores_
#         final_y_scores = model.y_scores_
#         transformed_source_data = th.matmul(source_data, final_x_scores)
#         transformed_target_data = th.matmul(target_data, final_y_scores)

#         transformed_source_data_np = transformed_source_data.numpy()
#         transformed_target_data_np = transformed_target_data.numpy()
#         source_label_np = source_label.numpy().ravel()
#         target_label_np = target_label.numpy().ravel()

#         accuracy, predicted_labels = Accuracy.knn_accuracy(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
#         auc_roc, f1_score_macro = Accuracy.roc_auc_f1(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
#         target_data_with_labels = target_data_df.copy()
#         target_data_with_labels['Predicted Label'] = predicted_labels
#         target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == predicted_labels, 'Correct', 'Incorrect')

#         target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')




#     elif model_type == 'dpls':
#         model = DeepPLS(
#             lv_dimensions=[30, 40],
#             pls_solver='iter',
#             use_nonlinear_mapping=False,
#             mapping_dimensions=[800, 800],
#             nys_gamma_values=[0.014, 2.8],
#             stack_previous_lv1=True
#         )
        
#         model.fit(source_data_transpose, target_data_transpose)
#         final_x_scores, final_y_scores = model.get_final_scores()
#         transformed_source_data = th.matmul(source_data, final_x_scores)
#         transformed_target_data = th.matmul(target_data, final_y_scores)

#         transformed_source_data_np = transformed_source_data.numpy()
#         transformed_target_data_np = transformed_target_data.numpy()
#         source_label_np = source_label.numpy().ravel()
#         target_label_np = target_label.numpy().ravel()

#         accuracy, predicted_labels = Accuracy.knn_accuracy(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
#         auc_roc, f1_score_macro = Accuracy.roc_auc_f1(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
#         target_data_with_labels = target_data_df.copy()
#         target_data_with_labels['Predicted Label'] = predicted_labels
#         target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == predicted_labels, 'Correct', 'Incorrect')

#         target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')


#     elif model_type == 'gdpls':
#         model = DeepPLS(
#             lv_dimensions=[30, 40],
#             pls_solver='iter',
#             use_nonlinear_mapping=True,
#             mapping_dimensions=[800, 800],
#             nys_gamma_values=[0.014, 2.8],
#             stack_previous_lv1=True
#         )
#         model.fit(source_data_transpose, target_data_transpose)
#         final_x_scores, final_y_scores = model.get_final_scores()
#         transformed_source_data = th.matmul(source_data, final_x_scores)
#         transformed_target_data = th.matmul(target_data, final_y_scores)

#         transformed_source_data_np = transformed_source_data.numpy()
#         transformed_target_data_np = transformed_target_data.numpy()
#         source_label_np = source_label.numpy().ravel()
#         target_label_np = target_label.numpy().ravel()

#         accuracy, predicted_labels = Accuracy.knn_accuracy(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
#         auc_roc, f1_score_macro = Accuracy.roc_auc_f1(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
#         target_data_with_labels = target_data_df.copy()
#         target_data_with_labels['Predicted Label'] = predicted_labels
#         target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == predicted_labels, 'Correct', 'Incorrect')
#         target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')

#     elif model_type == 'coral':
#         model = CORAL()
#         accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(source_data1,source_label1,target_data1,target_label1)
#         target_data_with_labels = target_data_df1.copy()
#         target_data_with_labels['Predicted Label'] = ypre
#         target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
#         target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')



#     elif model_type == 'tca':
#         model = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
#         accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(source_data1,source_label1,target_data1,target_label1)
#         target_data_with_labels = target_data_df1.copy()
#         target_data_with_labels['Predicted Label'] = ypre
#         target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
#         target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')

    
#     elif model_type == 'tcaPlus':
#         cS = TCA_PLUS.compute_characteristic_vector(source_data1)
#         cT = TCA_PLUS.compute_characteristic_vector(target_data1)
#         # Assign nominal values to similarity vector
#         similarity_vector = TCA_PLUS.assign_nominal_values(cS, cT)
#         # Apply normalization
#         Xs_norm, Xt_norm = TCA_PLUS.apply_normalization(source_data1, target_data1, similarity_vector)
        
#         model = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
#         accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(Xs_norm, source_label1, Xt_norm, target_label1)
#         target_data_with_labels = target_data_df1.copy()
#         target_data_with_labels['Predicted Label'] = ypre
#         target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
#         target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')



#     elif model_type == 'jda':
#         model = JDA(kernel_type='primal', dim=30, lamb=1, gamma=1)
#         accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(source_data1,source_label1,target_data1,target_label1)
#         target_data_with_labels = target_data_df1.copy()
#         target_data_with_labels['Predicted Label'] = ypre
#         target_data_with_labels['Defect or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
#         target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')

#     return jsonify({
#         'accuracy': accuracy,
#         'aucRoc': auc_roc,
#         'f1Score': f1_score_macro,
#         'targetDataWithLabels': target_data_with_labels_dict
#     })

# if __name__ == '__main__':
#     app.run(debug=True)








from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import sklearn
import torch as th
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from models import PLS, DeepPLS,Accuracy
from coral import CORAL
from tca import TCA
from jda import JDA
from bda import BDA
from tcaP import TCA_PLUS
from deepCoral import DeepCoralModel
from dtlc import DTLC

app = Flask(__name__)
CORS(app)


# def process_files(request):
#     if 'source' not in request.files or 'target' not in request.files:
#         return None, None, None, None, None, 'Source and target files are required'

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

#     source_data = (source_data - source_data.mean(dim=0)) / source_data.std(dim=0)
#     target_data = (target_data - target_data.mean(dim=0)) / target_data.std(dim=0)

#     return source_data, source_label, target_data, target_label, target_data_df, None




def process_files(request):
    if 'source' not in request.files or 'target' not in request.files:
        return None, None, None, None, None, 'Source and target files are required'

    source_file = request.files['source']
    target_file = request.files['target']

    try:
        # Determine file type for source and target
        if source_file.filename.endswith('.csv'):
            source_data_df = pd.read_csv(source_file)
        elif source_file.filename.endswith(('.xlsx', '.xls')):
            source_data_df = pd.read_excel(source_file)
        else:
            return None, None, None, None, None, f"Unsupported source file type: {source_file.filename}"

        if target_file.filename.endswith('.csv'):
            target_data_df = pd.read_csv(target_file)
        elif target_file.filename.endswith(('.xlsx', '.xls')):
            target_data_df = pd.read_excel(target_file)
        else:
            return None, None, None, None, None, f"Unsupported target file type: {target_file.filename}"

        # Extract data and labels
        source_data = th.tensor(source_data_df.iloc[:, :-1].values, dtype=th.float32)
        source_label = th.tensor(source_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

        target_data = th.tensor(target_data_df.iloc[:, :-1].values, dtype=th.float32)
        target_label = th.tensor(target_data_df.iloc[:, -1].values[:, None], dtype=th.float32)

        # Handle NaN and Inf values
        source_data[th.isnan(source_data)] = 0
        source_data[th.isinf(source_data)] = 0
        target_data[th.isnan(target_data)] = 0
        target_data[th.isinf(target_data)] = 0

        # Normalize data
        source_data = (source_data - source_data.mean(dim=0)) / source_data.std(dim=0)
        target_data = (target_data - target_data.mean(dim=0)) / target_data.std(dim=0)

        return source_data, source_label, target_data, target_label, target_data_df, None

    except Exception as e:
        print(f"Error while processing files: {e}")
        return None, None, None, None, None, f"Error while processing files: {e}"



def process_files1(request):
    if 'source' not in request.files or 'target' not in request.files:
        return None, None, None, None, None, 'Source and target files are required'

    source_file = request.files['source']
    target_file = request.files['target']

    try:
        # Determine file type
        if source_file.filename.endswith('.csv'):
            source_data_df = pd.read_csv(source_file)
        elif source_file.filename.endswith(('.xlsx', '.xls')):
            source_data_df = pd.read_excel(source_file)
        else:
            return None, None, None, None, None, f"Unsupported source file type: {source_file.filename}"

        if target_file.filename.endswith('.csv'):
            target_data_df = pd.read_csv(target_file)
        elif target_file.filename.endswith(('.xlsx', '.xls')):
            target_data_df = pd.read_excel(target_file)
        else:
            return None, None, None, None, None, f"Unsupported target file type: {target_file.filename}"

        # Extract data and labels
        source_data = source_data_df.iloc[:, :-1].values
        source_label = source_data_df.iloc[:, -1].values
        target_data = target_data_df.iloc[:, :-1].values
        target_label = target_data_df.iloc[:, -1].values

        return source_data, source_label, target_data, target_label, target_data_df, None

    except Exception as e:
        print(f"Error while processing files: {e}")
        return None, None, None, None, None, f"Error while processing files: {e}"




# def process_files1(request):
#     if 'source' not in request.files or 'target' not in request.files:
#         return None, None, None, None, None, 'Source and target files are required'

#     source_file = request.files['source']
#     target_file = request.files['target']
 
#     source_data_df = pd.read_excel(source_file)
#     target_data_df = pd.read_excel(target_file)
  
#     source_data = source_data_df.iloc[:, :-1].values
#     source_label = source_data_df.iloc[:, -1].values
#     target_data = target_data_df.iloc[:, :-1].values
#     target_label = target_data_df.iloc[:, -1].values

#     return source_data, source_label, target_data, target_label, target_data_df, None    

@app.route('/upload', methods=['POST'])
def upload_files():
    model_type = request.form.get('model_type')
    if model_type not in ['pls', 'dpls', 'gdpls', 'coral','tca','jda', 'bda', 'tcaPlus', 'deepCoral', 'dtlc']:
        return jsonify({'error': 'Invalid model type'}), 400

    source_data, source_label, target_data, target_label, target_data_df, error = process_files(request)
    if error:
        return jsonify({'error': error}), 400


    
    source_data1, source_label1, target_data1, target_label1, target_data_df1, error = process_files1(request)
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

        accuracy, predicted_labels = Accuracy.knn_accuracy(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        auc_roc, f1_score_macro = Accuracy.roc_auc_f1(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        target_data_with_labels = target_data_df.copy()
        target_data_with_labels['Predicted Label'] = predicted_labels
        target_data_with_labels['Correct or Not'] = np.where(target_data_with_labels.iloc[:, -2] == predicted_labels, 'Correct', 'Incorrect')

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

        accuracy, predicted_labels = Accuracy.knn_accuracy(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        auc_roc, f1_score_macro = Accuracy.roc_auc_f1(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        target_data_with_labels = target_data_df.copy()
        target_data_with_labels['Predicted Label'] = predicted_labels
        target_data_with_labels['Correct or Not'] = np.where(target_data_with_labels.iloc[:, -2] == predicted_labels, 'Correct', 'Incorrect')

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

        accuracy, predicted_labels = Accuracy.knn_accuracy(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        auc_roc, f1_score_macro = Accuracy.roc_auc_f1(transformed_source_data_np, source_label_np, transformed_target_data_np, target_label_np)
        target_data_with_labels = target_data_df.copy()
        target_data_with_labels['Predicted Label'] = predicted_labels
        target_data_with_labels['Correct or Not'] = np.where(target_data_with_labels.iloc[:, -2] == predicted_labels, 'Correct', 'Incorrect')
        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')

    elif model_type == 'coral':
        print(source_data1.dtype)
        print(source_data1)
        print('newwww')
        print(source_label1.dtype)
        print(source_label1)
        model = CORAL()
        accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(source_data1,source_label1,target_data1,target_label1)
        target_data_with_labels = target_data_df1.copy()
        target_data_with_labels['Predicted Label'] = ypre
        target_data_with_labels['Correct or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')

    elif model_type == 'deepCoral':
         model = DeepCoralModel(batch_size=64, num_epochs=100, learning_rate=0.001, n_neighbors=1)
         accuracy, ypre, auc_roc, f1_score_macro = model.fit_predict(source_data1, source_label1, target_data1, target_label1)
         target_data_with_labels = target_data_df1.copy()
         target_data_with_labels['Predicted Label'] = ypre
         target_data_with_labels['Correct or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
         target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')    
         

    elif model_type == 'tca':
        model = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
        accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(source_data1,source_label1,target_data1,target_label1)
        target_data_with_labels = target_data_df1.copy()
        target_data_with_labels['Predicted Label'] = ypre
        target_data_with_labels['Correct or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')

    
    elif model_type == 'tcaPlus':
        cS = TCA_PLUS.compute_characteristic_vector(source_data1)
        cT = TCA_PLUS.compute_characteristic_vector(target_data1)
        # Assign nominal values to similarity vector
        similarity_vector = TCA_PLUS.assign_nominal_values(cS, cT)
        # Apply normalization
        Xs_norm, Xt_norm = TCA_PLUS.apply_normalization(source_data1, target_data1, similarity_vector)
        
        model = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
        accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(Xs_norm, source_label1, Xt_norm, target_label1)
        target_data_with_labels = target_data_df1.copy()
        target_data_with_labels['Predicted Label'] = ypre
        target_data_with_labels['Correct or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')

    elif model_type == 'jda':
        model = JDA(kernel_type='primal', dim=30, lamb=1, gamma=1)
        accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(source_data1,source_label1,target_data1,target_label1)
        target_data_with_labels = target_data_df1.copy()
        target_data_with_labels['Predicted Label'] = ypre
        target_data_with_labels['Correct or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')


    elif model_type == 'bda':
        model = BDA(kernel_type='primal', dim=30, lamb=1, gamma=1)
        accuracy,ypre, auc_roc, f1_score_macro=model.fit_predict(source_data1,source_label1,target_data1,target_label1)
        target_data_with_labels = target_data_df1.copy()
        target_data_with_labels['Predicted Label'] = ypre
        target_data_with_labels['Correct or Not'] = np.where(target_data_with_labels.iloc[:, -2] == ypre, 'Correct', 'Incorrect')
        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')    



     # In app.py:
    elif model_type == 'dtlc':
        model = DTLC(dim=30, max_iter=10, alpha=1.0, beta=1.0, eta=1.0)
        accuracy, ypre, auc_roc, f1_score_macro = model.fit_predict(
            source_data1, 
            source_label1,
            target_data1,
            target_label1
        )
        target_data_with_labels = target_data_df1.copy()
        target_data_with_labels['Predicted Label'] = ypre
        target_data_with_labels['Correct or Not'] = np.where(
            target_data_with_labels.iloc[:, -2] == ypre, 
            'Correct', 
            'Incorrect'
        )
        target_data_with_labels_dict = target_data_with_labels.to_dict(orient='records')   

    return jsonify({
        'accuracy': f"{accuracy * 100:.2f}%",
        'aucRoc': f"{auc_roc:.3f}",
        'f1Score': f"{f1_score_macro:.3f}",
        'targetDataWithLabels': target_data_with_labels_dict
    })

if __name__ == '__main__':
    app.run(debug=True)
