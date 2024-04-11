from src.core.constants import MONGO_DB_CONSTANTS, FEATURE_FORMAT_CONSTANTS
from src.core.ml.nlp_features_utils import NlpFeaturesUtils
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os
import json
import csv

class MlModelsUtils : 
    
    @staticmethod
    def execute_train_test(train_path, test_path, model, save_folder, model_name, one_hot_encode = False) :
        """
        Execute training and testing of a machine learning model and save the model, it's predictions for the test set and the train and test classification reports.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.
            model: The machine learning model to train and test.
            save_folder (str): Folder path to save the trained model and reports.
            model_name (str): Name of the model for saving purposes.
            one_hot_encode (bool, optional): Whether to perform one-hot encoding. Default is False.
        """
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train, y_train = train_df.drop(columns=[MONGO_DB_CONSTANTS.TONE_FIELD,  MONGO_DB_CONSTANTS.ID_FIELD]), train_df[MONGO_DB_CONSTANTS.TONE_FIELD]
        X_test, y_test = test_df.drop(columns=[MONGO_DB_CONSTANTS.TONE_FIELD,  MONGO_DB_CONSTANTS.ID_FIELD]), test_df[MONGO_DB_CONSTANTS.TONE_FIELD]

        encoder = LabelBinarizer() if one_hot_encode else LabelEncoder()
        
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(save_folder, model_name + '.pkl'))

        train_predictions = model.predict(X_train)
        train_classification_report = classification_report(train_predictions, y_train)
        train_classification_file_report = open(os.path.join(save_folder, 'train_report.txt'), 'w+')
        train_classification_file_report.write(train_classification_report)
        train_classification_file_report.close()


        test_preds = model.predict(X_test)
        test_classification_report = classification_report(y_test, test_preds)
        test_classification_report_file = open(os.path.join(save_folder, 'test_report.txt'), 'w+')
        test_classification_report_file.write(test_classification_report)
        test_classification_report_file.close()
  
        with open(os.path.join(save_folder, 'predictions.csv'), 'w+') as predictions_file : 
            writer = csv.writer(predictions_file)
            writer.writerow(['_id', 'label', 'prediction'])

            for pred_idx, prediction in enumerate(test_preds) : 
                writer.writerow([test_df.iloc[pred_idx][MONGO_DB_CONSTANTS.ID_FIELD],y_test[pred_idx], prediction])

        predictions_file.close()


    
    
    @staticmethod
    def use_trained_model_to_preidct_tone(model_path, input_text, **kwargs) :
        """
        Uses a trained machine learning model to predict the tone of the given input text.

        Parameters:
        - model_path: The path to the trained machine learning model file.
        - input_text: The input text for which to predict the tone.
        - **kwargs: Additional keyword arguments.
            - feature_vector_format: The format of the feature vectors ('bow', 'tf-idf', 'w2v_max', 'w2v_sum', 'w2v_mean').
            - tf-idf-data-path: The path to the TF-IDF data file (required for 'bow' or 'tf-idf' feature formats).
            - w2v_model_path: The path to the Word2Vec model file (required for Word2Vec feature formats).

        Returns:
        - prediction: The predicted tone of the input text.
        """

        model = joblib.load(model_path)
        
        feature_vector_format = kwargs['feature_vector_format']

        if feature_vector_format == 'bow' or feature_vector_format == FEATURE_FORMAT_CONSTANTS.TF_IDF :
            
            bow_feature_vector, tf_idf_feature_vector = NlpFeaturesUtils.generate_bow_tf_idf_feature_vector(input_text, None, kwargs['tf-idf-data-path'])

            if feature_vector_format == FEATURE_FORMAT_CONSTANTS.BOW:
                prediction = model.predict(bow_feature_vector)
                return prediction
            return model.predict(tf_idf_feature_vector)    
        
        max_dims, sum_dims, mean_dims = NlpFeaturesUtils.generate_word2vec_feature_vector(kwargs['w2v_model_path'], input_text)
        if feature_vector_format == FEATURE_FORMAT_CONSTANTS.W2V_MAX:
            return model.predict(max_dims)
        if feature_vector_format == FEATURE_FORMAT_CONSTANTS.W2V_SUM :
            return sum_dims
        return mean_dims









    

