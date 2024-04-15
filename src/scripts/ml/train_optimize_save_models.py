from src.core.ml.ml_models_utils import MlModelsUtils
from src.core.constants import FEATURE_FORMAT_CONSTANTS
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import csv
import threading
import uuid




# Locks for ensuring thread-safe access to shared resources
feature_vector_format_folder_lock = threading.Lock()
writer_lock = threading.Lock()
max_threads = threading.active_count()

def train_test_save_model(n_split, dataset_folder, output_folder, model, feature_vector_format,  model_uuid) :
    """
    Train, test, and save a machine learning model for each split of a dataset.

    Args:
        n_split (int): Number of splits in the dataset.
        dataset_folder (str): Folder path containing the dataset splits.
        output_folder (str): Folder path to save the trained models and reports.
        model: The machine learning model to train and test.
        feature_vector_format (str): Format of the feature vectors.

    Returns:
        - model_folder_path (Path) : The path containing the split sub-folder
        - split_test_accuracies (list[n_split]): The accuracy of the model on the test set of each split.
    """
    feature_vector_format_folder = os.path.join(output_folder, feature_vector_format)

    # Acquire lock to ensure safe creation of feature vector format folder
    feature_vector_format_folder_lock.acquire()
    if not os.path.exists(feature_vector_format_folder) :
            os.mkdir(feature_vector_format_folder)
    
    feature_vector_format_folder_lock.release()

    model_folder_path = os.path.join(feature_vector_format_folder, f'{model.__class__.__name__}({model_uuid})')
    os.mkdir(model_folder_path) 

    split_test_accuracies = []
    for split in range(n_split) :
        save_folder_path = os.path.join(model_folder_path, f'split{split}')
        os.mkdir(save_folder_path)

        data_split_path = os.path.join(dataset_folder,  f'split{split}')

        train_path = os.path.join(data_split_path, feature_vector_format + '_train.csv')
        test_path = os.path.join(data_split_path, feature_vector_format + '_test.csv')
        
        split_test_accuracy = MlModelsUtils.execute_train_test(train_path, test_path, model, save_folder_path)
        split_test_accuracies.append(split_test_accuracy)
    
    return model_folder_path, split_test_accuracies

def hyper_params_search_train_test_save(n_split, dataset_folder, output_folder, model_constructor, params_list, writer) :
    """
    Trains a model with different hyperparameters, evaluates their performance on test data, and saves the trained models along with their evaluation results.

    Parameters:
    - n_split: The number of splits for cross-validation.
    - dataset_folder: The folder containing the dataset.
    - output_folder: The folder where the trained models and evaluation results will be saved.
    - model_constructor: The constructor function for creating the model.
    - params_list: A list of dictionaries, each containing hyperparameters for model construction.
    - writer: CSV writer object for writing the training data.

    Returns:
    - training_data: A list of lists, each containing information about the trained models and their evaluation results.
                     Each inner list contains:
                     - Model name
                     - Feature vector format
                     - Test accuracy
                     - Path to the saved model
                     - Split index
    """
    training_data = []
    for params in params_list :
        model_param_config_uid  = uuid.uuid4()
        try :
            for feature_vector_format in FEATURE_FORMAT_CONSTANTS.FEATURES_NAMES:
                model = model_constructor(**params)
                model_folder_path, split_test_accuracies = train_test_save_model(n_split, dataset_folder, output_folder, model, feature_vector_format, model_param_config_uid)
                
                for split_idx in range(len(split_test_accuracies)) :
                    accuracy = split_test_accuracies[split_idx]
                    model_path = os.path.join(model_folder_path, f'split{split_idx}', f"{str(model)}.pkl")
                    training_data.append([str(model).split('(')[0], model_param_config_uid, feature_vector_format, accuracy, model_path, split_idx])
        except Exception as e : 
            print(e)

    writer_lock.acquire()
    writer.writerows(training_data)
    writer_lock.release()
    return training_data

if __name__ == "__main__" :
    """
    Train multiple models concurrently using separate threads with different hyperparameters, 
    evaluate their performance on test data, and save the trained models 
    along with their evaluation results to a CSV file.

    Parameters:
    - dataset_folder: The folder containing the dataset.
    - output_folder: The folder where the trained models and evaluation results will be saved.
    - summary_file_path: The path to the CSV file where the summary of trained models' data will be saved, this file is necesary for the flask Server.
    - n_split: The number of splits for cross-validation.
    """

    dataset_folder = 'datasets'
    output_folder = 'models_training'
    summary_file_path = 'trained_models_data.csv'
    n_split = 3

    file = open(summary_file_path, 'w+', newline='') 
    writer = csv.writer(file)
    columns_names = ['model_name','param_uuid' 'feature_format', 'accuracy', 'path', 'split']
    writer.writerow(columns_names)

    rf_params = [{'n_estimators': 100, 'max_depth': None, 'criterion': 'gini', 'min_samples_split': 2, 'min_samples_leaf': 1},
             {'n_estimators': 200, 'max_depth': 50, 'criterion': 'entropy', 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
             {'n_estimators': 150, 'max_depth': 100, 'criterion': 'gini', 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'log2'},
             {'n_estimators': 150, 'max_depth': 150, 'criterion': 'entropy', 'min_samples_split': 20, 'min_samples_leaf': 10},
             {'n_estimators': 100, 'max_depth': 200, 'criterion': 'gini', 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
             {'n_estimators': 200, 'max_depth': None, 'criterion': 'entropy', 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2'},
             {'n_estimators': 200, 'max_depth': 100, 'criterion': 'gini', 'min_samples_split': 10, 'min_samples_leaf': 5},
             {'n_estimators': 100, 'max_depth': 50, 'criterion': 'entropy', 'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'sqrt'},
             {'n_estimators': 150, 'max_depth': 150, 'criterion': 'gini', 'min_samples_split': 5, 'min_samples_leaf': 2},
             {'n_estimators': 100, 'max_depth': None, 'criterion': 'entropy', 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
             {'n_estimators': 200, 'max_depth': 50, 'criterion': 'gini', 'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'log2'},
             {'n_estimators': 150, 'max_depth': 100, 'criterion': 'entropy', 'min_samples_split': 2, 'min_samples_leaf': 1},
             {'n_estimators': 100, 'max_depth': 200, 'criterion': 'gini', 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
             {'n_estimators': 200, 'max_depth': None, 'criterion': 'entropy', 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'log2'},
             {'n_estimators': 150, 'max_depth': 50, 'criterion': 'gini', 'min_samples_split': 20, 'min_samples_leaf': 10},
             {'n_estimators': 100, 'max_depth': 150, 'criterion': 'entropy', 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
             {'n_estimators': 200, 'max_depth': 200, 'criterion': 'gini', 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2'},
             {'n_estimators': 150, 'max_depth': None, 'criterion': 'entropy', 'min_samples_split': 10, 'min_samples_leaf': 5},
             {'n_estimators': 100, 'max_depth': 50, 'criterion': 'gini', 'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'sqrt'},
             
             {'n_estimators': 100, 'max_depth': None, 'criterion': 'gini', 'min_samples_split': 2, 'min_samples_leaf': 1, 'class_weight' : 'balanced'},
             {'n_estimators': 200, 'max_depth': 50, 'criterion': 'entropy', 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight' : 'balanced'},
             {'n_estimators': 150, 'max_depth': 100, 'criterion': 'gini', 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'log2', 'class_weight' : 'balanced'},
             {'n_estimators': 150, 'max_depth': 150, 'criterion': 'entropy', 'min_samples_split': 20, 'min_samples_leaf': 10, 'class_weight' : 'balanced'},
             {'n_estimators': 100, 'max_depth': 200, 'criterion': 'gini', 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight' : 'balanced'},
             {'n_estimators': 200, 'max_depth': None, 'criterion': 'entropy', 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'class_weight' : 'balanced'},
             {'n_estimators': 200, 'max_depth': 100, 'criterion': 'gini', 'min_samples_split': 10, 'min_samples_leaf': 5, 'class_weight' : 'balanced'},
             {'n_estimators': 100, 'max_depth': 50, 'criterion': 'entropy', 'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'class_weight' : 'balanced'},
             {'n_estimators': 150, 'max_depth': 150, 'criterion': 'gini', 'min_samples_split': 5, 'min_samples_leaf': 2, 'class_weight' : 'balanced'},
             {'n_estimators': 100, 'max_depth': None, 'criterion': 'entropy', 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'class_weight' : 'balanced'},
             {'n_estimators': 200, 'max_depth': 50, 'criterion': 'gini', 'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'log2', 'class_weight' : 'balanced'},
             {'n_estimators': 150, 'max_depth': 100, 'criterion': 'entropy', 'min_samples_split': 2, 'min_samples_leaf': 1, 'class_weight' : 'balanced'},
             {'n_estimators': 100, 'max_depth': 200, 'criterion': 'gini', 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight' : 'balanced'},
             {'n_estimators': 200, 'max_depth': None, 'criterion': 'entropy', 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'log2', 'class_weight' : 'balanced'},
             {'n_estimators': 150, 'max_depth': 50, 'criterion': 'gini', 'min_samples_split': 20, 'min_samples_leaf': 10, 'class_weight' : 'balanced'},
             {'n_estimators': 100, 'max_depth': 150, 'criterion': 'entropy', 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight' : 'balanced'},
             {'n_estimators': 200, 'max_depth': 200, 'criterion': 'gini', 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'class_weight' : 'balanced'},
             {'n_estimators': 150, 'max_depth': None, 'criterion': 'entropy', 'min_samples_split': 10, 'min_samples_leaf': 5, 'class_weight' : 'balanced'},
             {'n_estimators': 100, 'max_depth': 50, 'criterion': 'gini', 'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'class_weight' : 'balanced'}]

    
    

    ada_params = [{'n_estimators': 50, 'learning_rate': 1.0},
              {'n_estimators': 100, 'learning_rate': 1.0},
              {'n_estimators': 200, 'learning_rate': 1.0},
              {'n_estimators': 50, 'learning_rate': 0.5},
              {'n_estimators': 100, 'learning_rate': 0.5},
              {'n_estimators': 200, 'learning_rate': 0.5},
              {'n_estimators': 50, 'learning_rate': 0.1},
              {'n_estimators': 100, 'learning_rate': 0.1},
              {'n_estimators': 200, 'learning_rate': 0.1}]
    

    gb_params = [{'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 1.0, 'max_features': None},
             {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 1.0, 'max_features': None},
             {'learning_rate': 0.01, 'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 1.0, 'max_features': None},
             {'learning_rate': 0.1, 'n_estimators': 150, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 1.0, 'max_features': None},
             {'learning_rate': 0.05, 'n_estimators': 150, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 1.0, 'max_features': None},
             {'learning_rate': 0.01, 'n_estimators': 150, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 1.0, 'max_features': None},
             {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 1.0, 'max_features': None},
             {'learning_rate': 0.05, 'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 1.0, 'max_features': None},
             {'learning_rate': 0.01, 'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 1.0, 'max_features': None}]

    
    
    hgb_params = [
    {'learning_rate': 0.1, 'max_iter': 100, 'max_depth': 3},
    {'learning_rate': 0.05, 'max_iter': 100, 'max_depth': 3},
    {'learning_rate': 0.01, 'max_iter': 100, 'max_depth': 3},
    {'learning_rate': 0.1, 'max_iter': 150, 'max_depth': 3},
    {'learning_rate': 0.05, 'max_iter': 150, 'max_depth': 3},
    {'learning_rate': 0.01, 'max_iter': 150, 'max_depth': 3},
    {'learning_rate': 0.1, 'max_iter': 200, 'max_depth': 3},
    {'learning_rate': 0.05, 'max_iter': 200, 'max_depth': 3},
    {'learning_rate': 0.01, 'max_iter': 200, 'max_depth': 3},

    {'learning_rate': 0.1, 'max_iter': 100, 'max_depth': 3, 'class_weight':'balanced'},
    {'learning_rate': 0.05, 'max_iter': 100, 'max_depth': 3, 'class_weight':'balanced'},
    {'learning_rate': 0.01, 'max_iter': 100, 'max_depth': 3, 'class_weight':'balanced'},
    {'learning_rate': 0.1, 'max_iter': 150, 'max_depth': 3, 'class_weight':'balanced'},
    {'learning_rate': 0.05, 'max_iter': 150, 'max_depth': 3, 'class_weight':'balanced'},
    {'learning_rate': 0.01, 'max_iter': 150, 'max_depth': 3, 'class_weight':'balanced'},
    {'learning_rate': 0.1, 'max_iter': 200, 'max_depth': 3, 'class_weight':'balanced'},
    {'learning_rate': 0.05, 'max_iter': 200, 'max_depth': 3, 'class_weight':'balanced'},
    {'learning_rate': 0.01, 'max_iter': 200, 'max_depth': 3, 'class_weight':'balanced'}
    ]


    mlp_params = [
    {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (50, 50), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (100, 100), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (50, 50, 50), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (100, 50, 25), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (100, 100, 50), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (200, 100, 50), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (200, 200, 100), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001}
]


    #search_params = [(RandomForestClassifier, rf_params),(AdaBoostClassifier, ada_params), (GradientBoostingClassifier, gb_params),(HistGradientBoostingClassifier, hgb_params),(MLPClassifier, mlp_params)]
    #threads = []
    search_params = [(MLPClassifier, [{}])]
    # Start a thread for each model
    for model, params in search_params :
        
        hyper_params_search_train_test_save(n_split,dataset_folder, output_folder, model, params, writer)
        #thread = threading.Thread(target=hyper_params_search_train_test_save, args=(n_split,dataset_folder, output_folder, model, params, writer))
        #threads.append(thread)
    

    file.close()
        

    
    # Wait for all threads to finish
    #for thread in threads :
    #    thread.join()
         


