from src.core.ml_utils.ml_utils import MlUtils
from src.core.constants import FEATURE_FORMAT_CONSTANTS
import os
import sklearn


if __name__ == "__main__" :
    dataset_folder = 'datasets'
    output_folder = 'output_folder'
    k = 5

    
    for split in range(k) :
        split_path = os.path.join(dataset_folder, f'split{split}')

        for feature_vector_format in [FEATURE_FORMAT_CONSTANTS.BOW, FEATURE_FORMAT_CONSTANTS.TF_IDF, FEATURE_FORMAT_CONSTANTS.W2V_MAX, FEATURE_FORMAT_CONSTANTS.W2V_SUM, FEATURE_FORMAT_CONSTANTS.W2V_MEAN] : 
            feature_vector_format_folder = os.path.join(output_folder, feature_vector_format)
            if not os.path.exists(feature_vector_format_folder) :
                os.mkdir(feature_vector_format_folder)

            
            train_path = os.path.join(split_path, feature_vector_format + '_train.csv')
            test_path = os.path.join(split_path, feature_vector_format + '_test.csv')
            save_folder_path = os.path.join(feature_vector_format_folder, str(model))
            save_folder = os.mkdir(save_folder_path)
            MlUtils.execute_train_test(train_path, test_path, model, save_folder_path, str(model))