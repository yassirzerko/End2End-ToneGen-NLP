from src.core.ml.ml_models_utils import MlModelsUtils
from src.core.constants import FEATURE_FORMAT_CONSTANTS
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def train_test_save_model(n_split, dataset_folder, output_folder, model, feature_vector_format, one_hot_encode) :
    """
    Train, test, and save a machine learning model for each split of a dataset.

    Args:
        n_split (int): Number of splits in the dataset.
        dataset_folder (str): Folder path containing the dataset splits.
        output_folder (str): Folder path to save the trained models and reports.
        model: The machine learning model to train and test.
        feature_vector_format (str): Format of the feature vectors.
    """
    feature_vector_format_folder = os.path.join(output_folder, feature_vector_format)
    if not os.path.exists(feature_vector_format_folder) :
            os.mkdir(feature_vector_format_folder)

    model_folder_path = os.path.join(feature_vector_format_folder, str(model))
    os.mkdir(model_folder_path) 

    for split in range(n_split) :
        save_folder_path = os.path.join(model_folder_path, f'split{split}')
        os.mkdir(save_folder_path)

        data_split_path = os.path.join(dataset_folder,  f'split{split}')

        train_path = os.path.join(data_split_path, feature_vector_format + '_train.csv')
        test_path = os.path.join(data_split_path, feature_vector_format + '_test.csv')
        
        MlModelsUtils.execute_train_test(train_path, test_path, model, save_folder_path, str(model), one_hot_encode)

if __name__ == "__main__" :
    '''
    This script trains and tests and save machine learning models with different parameters and feature vector formats.
    '''

    dataset_folder = 'datasets'
    output_folder = 'output_folder'
    n_split = 3

    
    rf_params = [{'n_estimators' : 100, 'max_depth' : None, 'criterion' : 'gini'}, {'n_estimators' : 100, 'max_depth' : 200, 'criterion' : 'gini'} , {'n_estimators' : 100, 'max_depth' : 100, 'criterion' : 'gini'},
              {'n_estimators' : 100, 'max_depth' : 50, 'criterion' : 'gini'},{'n_estimators' : 50, 'max_depth' : None, 'criterion' : 'gini'}, {'n_estimators' : 50, 'max_depth' : 200, 'criterion' : 'gini'},
              {'n_estimators' : 50, 'max_depth' : 100, 'criterion' : 'gini'}, {'n_estimators' : 50, 'max_depth' : 50, 'criterion' : 'gini'}]

    
    for params in rf_params : 
        for feature_vector_format in [FEATURE_FORMAT_CONSTANTS.BOW, FEATURE_FORMAT_CONSTANTS.TF_IDF, FEATURE_FORMAT_CONSTANTS.W2V_MAX, FEATURE_FORMAT_CONSTANTS.W2V_SUM, FEATURE_FORMAT_CONSTANTS.W2V_MEAN] :
            rf = RandomForestClassifier(**params)
            train_test_save_model(n_split, dataset_folder, output_folder, rf, feature_vector_format, False)


    ada_params = [{'n_estimators':200, 'learning_rate':1.0}, {'n_estimators':100, 'learning_rate':1.0}, {'n_estimators':50, 'learning_rate':1.0}, {'n_estimators':25, 'learning_rate':1.0},
                  {'n_estimators':200, 'learning_rate':0.5}, {'n_estimators':100, 'learning_rate':0.5}, {'n_estimators':50, 'learning_rate':0.5}, {'n_estimators':25, 'learning_rate':0.5},
                  {'n_estimators':200, 'learning_rate':0.1}, {'n_estimators':100, 'learning_rate':0.1}, {'n_estimators':50, 'learning_rate':0.1}, {'n_estimators':25, 'learning_rate':0.1}]

    for params in ada_params : 
        for feature_vector_format in [FEATURE_FORMAT_CONSTANTS.BOW, FEATURE_FORMAT_CONSTANTS.TF_IDF, FEATURE_FORMAT_CONSTANTS.W2V_MAX, FEATURE_FORMAT_CONSTANTS.W2V_SUM, FEATURE_FORMAT_CONSTANTS.W2V_MEAN] :
            ada = AdaBoostClassifier(**params)
            train_test_save_model(n_split, dataset_folder, output_folder, ada, feature_vector_format, False)
    

    

    gb_params = [{'learning_rate':0.1, 'n_estimators':100}, {'learning_rate':0.05, 'n_estimators':100}, {'learning_rate':0.01, 'n_estimators':100}, 
                 {'learning_rate':0.1, 'n_estimators':150}, {'learning_rate':0.05, 'n_estimators':150}, {'learning_rate':0.01, 'n_estimators':150},
                 {'learning_rate':0.1, 'n_estimators':200}, {'learning_rate':0.05, 'n_estimators':200}, {'learning_rate':0.01, 'n_estimators':200}]

    for params in gb_params : 
        for feature_vector_format in [FEATURE_FORMAT_CONSTANTS.BOW, FEATURE_FORMAT_CONSTANTS.TF_IDF, FEATURE_FORMAT_CONSTANTS.W2V_MAX, FEATURE_FORMAT_CONSTANTS.W2V_SUM, FEATURE_FORMAT_CONSTANTS.W2V_MEAN] :
            gb = GradientBoostingClassifier(**params)
            train_test_save_model(n_split, dataset_folder, output_folder, gb, feature_vector_format, False)

    
    for feature_vector_format in [FEATURE_FORMAT_CONSTANTS.BOW, FEATURE_FORMAT_CONSTANTS.TF_IDF, FEATURE_FORMAT_CONSTANTS.W2V_MAX, FEATURE_FORMAT_CONSTANTS.W2V_SUM, FEATURE_FORMAT_CONSTANTS.W2V_MEAN] :
        hgb = HistGradientBoostingClassifier()
        train_test_save_model(n_split, dataset_folder, output_folder, hgb, feature_vector_format, False)

    mlp_params = [{'hidden_layer_size' : (100,100)}, {'hidden_layer_size' : (100,100, 100)}, {'hidden_layer_size' : (100,5,5)}, {'hidden_layer_size' : (25,25,25,25,25)} ]
    for params in mlp_params :
        for feature_vector_format in [FEATURE_FORMAT_CONSTANTS.BOW, FEATURE_FORMAT_CONSTANTS.TF_IDF, FEATURE_FORMAT_CONSTANTS.W2V_MAX, FEATURE_FORMAT_CONSTANTS.W2V_SUM, FEATURE_FORMAT_CONSTANTS.W2V_MEAN] :
            mlp = MLPClassifier(**params)
            train_test_save_model(n_split, dataset_folder, output_folder, mlp, feature_vector_format, True)
    


