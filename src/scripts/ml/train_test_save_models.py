from src.core.ml.ml_models_utils import MlModelsUtils
from src.core.constants import FEATURE_FORMAT_CONSTANTS
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import csv
import re


def train_test_save_model(n_split, dataset_folder, output_folder, model, feature_vector_format) :
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
    if not os.path.exists(feature_vector_format_folder) :
            os.mkdir(feature_vector_format_folder)

    model_folder_path = os.path.join(feature_vector_format_folder, str(model))
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

def hyper_params_train_test_save(n_split, dataset_folder, output_folder, model_constructor, params_list) :
    """
    Trains a model with different hyperparameters, evaluates their performance on test data, and saves the trained models along with their evaluation results.

    Parameters:
    - n_split: The number of splits for cross-validation.
    - dataset_folder: The folder containing the dataset.
    - output_folder: The folder where the trained models and evaluation results will be saved.
    - model_constructor: The constructor function for creating the model.
    - params_list: A list of dictionaries, each containing hyperparameters for model construction.

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
        for feature_vector_format in [FEATURE_FORMAT_CONSTANTS.BOW, FEATURE_FORMAT_CONSTANTS.TF_IDF, FEATURE_FORMAT_CONSTANTS.W2V_MAX, FEATURE_FORMAT_CONSTANTS.W2V_SUM, FEATURE_FORMAT_CONSTANTS.W2V_MEAN] :
            model = model_constructor(**params)
            model_folder_path, split_test_accuracies = train_test_save_model(n_split, dataset_folder, output_folder, model, feature_vector_format)
            
            for split_idx in range(len(split_test_accuracies)) :
                accuracy = split_test_accuracies[split_idx]
                model_path = os.path.join(model_folder_path, f'split{split_idx}.pkl')
                training_data.append([re.sub(r'\([^()]*\)', '', str(model)), feature_vector_format, accuracy, model_path, split_idx])

    return training_data

if __name__ == "__main__" :
    """
    Train multiple models with different hyperparameters using cross-validation, evaluate their performance on test data, and save the trained models along with their evaluation results to a CSV file.

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

    file = open(summary_file_path, 'w+') 
    writer = csv.writer(file)
    columns_names = ['model_name', 'feature_format', 'accuracy', 'path', 'split']
    writer.writerow(columns_names)

    rf_params = [{'n_estimators' : 100, 'max_depth' : None, 'criterion' : 'gini'}, {'n_estimators' : 100, 'max_depth' : 200, 'criterion' : 'gini'} , {'n_estimators' : 100, 'max_depth' : 100, 'criterion' : 'gini'},
              {'n_estimators' : 100, 'max_depth' : 50, 'criterion' : 'gini'},{'n_estimators' : 50, 'max_depth' : None, 'criterion' : 'gini'}, {'n_estimators' : 50, 'max_depth' : 200, 'criterion' : 'gini'},
              {'n_estimators' : 50, 'max_depth' : 100, 'criterion' : 'gini'}, {'n_estimators' : 50, 'max_depth' : 50, 'criterion' : 'gini'}]
    

    ada_params = [{'n_estimators':200, 'learning_rate':1.0}, {'n_estimators':100, 'learning_rate':1.0}, {'n_estimators':50, 'learning_rate':1.0}, {'n_estimators':25, 'learning_rate':1.0},
                  {'n_estimators':200, 'learning_rate':0.5}, {'n_estimators':100, 'learning_rate':0.5}, {'n_estimators':50, 'learning_rate':0.5}, {'n_estimators':25, 'learning_rate':0.5},
                  {'n_estimators':200, 'learning_rate':0.1}, {'n_estimators':100, 'learning_rate':0.1}, {'n_estimators':50, 'learning_rate':0.1}, {'n_estimators':25, 'learning_rate':0.1}]
    

    gb_params = [{'learning_rate':0.1, 'n_estimators':100}, {'learning_rate':0.05, 'n_estimators':100}, {'learning_rate':0.01, 'n_estimators':100}, 
                 {'learning_rate':0.1, 'n_estimators':150}, {'learning_rate':0.05, 'n_estimators':150}, {'learning_rate':0.01, 'n_estimators':150},
                 {'learning_rate':0.1, 'n_estimators':200}, {'learning_rate':0.05, 'n_estimators':200}, {'learning_rate':0.01, 'n_estimators':200}]
    
    hgb_params = [{
        'loss':'log_loss', 'learning_rate':0.1
    }]

    mlp_params = [{'hidden_layer_sizes' : (5,5)},{'hidden_layer_sizes' : (10,10)},{'hidden_layer_sizes' : (10,10,10)}, {'hidden_layer_sizes' : (100,100)}, {'hidden_layer_sizes' : (100,100, 100)}, {'hidden_layer_sizes' : (100,5,5)}, {'hidden_layer_sizes' : (25,25,25,25,25)} ]


    writer.writerows(hyper_params_train_test_save(n_split, dataset_folder, output_folder, RandomForestClassifier, rf_params))
    writer.writerows(hyper_params_train_test_save(n_split, dataset_folder, output_folder, AdaBoostClassifier, ada_params))
    writer.writerows(hyper_params_train_test_save(n_split, dataset_folder, output_folder, GradientBoostingClassifier, gb_params))    
    writer.writerows(hyper_params_train_test_save(n_split, dataset_folder, output_folder, HistGradientBoostingClassifier, hgb_params))
    writer.writerows(hyper_params_train_test_save(n_split, dataset_folder, output_folder, MLPClassifier, mlp_params))
   
    writer.close()


