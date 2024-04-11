from src.core.constants import MONGO_DB_CONSTANTS, FEATURE_FORMAT_CONSTANTS, TONES_CONSTANTS
from src.core.ml.nlp_features_utils import NlpFeaturesUtils
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import csv
import torch.nn as nn
from sklearn.neural_network import MLPClassifier

class MlModelsUtils : 
    
    @staticmethod
    def execute_train_test(train_path, test_path, model, save_folder) :
        """
        Execute training and testing of a machine learning model and save the model, it's predictions for the test set and the train and test classification reports.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.
            model: The machine learning model to train and test.
            save_folder (str): Folder path to save the trained model and reports.
        
        Returns:
        - test_accuracy (float): The accuracy of the model on the test set.
        """
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train, y_train = train_df.drop(columns=[MONGO_DB_CONSTANTS.TONE_FIELD,  MONGO_DB_CONSTANTS.ID_FIELD]), train_df[MONGO_DB_CONSTANTS.TONE_FIELD]
        X_test, y_test = test_df.drop(columns=[MONGO_DB_CONSTANTS.TONE_FIELD,  MONGO_DB_CONSTANTS.ID_FIELD]), test_df[MONGO_DB_CONSTANTS.TONE_FIELD]

        encoder = LabelBinarizer() if isinstance(model, MLPClassifier) or isinstance(model, nn.Module) else LabelEncoder()
        
        encoder.fit(TONES_CONSTANTS.TONES)
        y_train = encoder.transform(y_train)
        y_test = encoder.transform(y_test)

        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(save_folder, str(model) + '.pkl'))

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

        test_accuracy = accuracy_score(test_preds, y_test)
        return test_accuracy

    @staticmethod
    def use_trained_model_to_preidct_tone(model_path, input_text, **kwargs) :
        """
        Uses a trained machine learning model to predict the tone of the given input text.

        Parameters:
        - model_path: The path to the trained machine learning model file.
        - input_text: The input text for which to predict the tone.
        - **kwargs: Additional keyword arguments.
            - feature_vector_format: The format of the feature vectors ('bow', 'tf-idf', 'w2v_max', 'w2v_sum', 'w2v_mean').
            - idf_data_path: The path to the IDF data file (required for 'bow' or 'tf-idf' feature formats).
            - w2v_model_path: The path to the Word2Vec model file (required for Word2Vec feature formats).

        Returns:
        - prediction: The predicted tone of the input text.
        """
        feature_vector_format = kwargs['feature_vector_format']

        model = joblib.load(model_path)

        encoder = LabelBinarizer() if isinstance(model, MLPClassifier) or isinstance(model, nn.Module) else LabelEncoder()
        
        encoder.fit(TONES_CONSTANTS.TONES)

        prediction = None

        if feature_vector_format == FEATURE_FORMAT_CONSTANTS.BOW or feature_vector_format == FEATURE_FORMAT_CONSTANTS.TF_IDF :
            
            bow_feature_vector, tf_idf_feature_vector = NlpFeaturesUtils.generate_bow_tf_idf_feature_vector(input_text, None, kwargs['idf_data_path'])

            if feature_vector_format == FEATURE_FORMAT_CONSTANTS.BOW : 
                prediction = model.predict(bow_feature_vector)
            else : 
                prediction = model.predict(tf_idf_feature_vector)
        
        else :

            max_dims, sum_dims, mean_dims = NlpFeaturesUtils.generate_word2vec_feature_vector(kwargs['w2v_model_path'], input_text)
        
            if feature_vector_format == FEATURE_FORMAT_CONSTANTS.W2V_MAX:
                prediction =  model.predict(max_dims)
            
            elif feature_vector_format == FEATURE_FORMAT_CONSTANTS.W2V_SUM :
                prediction = model.predict(sum_dims)
            
            else : 
                prediction = model.prediction(mean_dims)
        
        return encoder.inverse_transform(prediction)









    

