from src.core.constants import MONGO_DB_CONSTANTS, FEATURE_FORMAT_CONSTANTS, TONES_CONSTANTS, W2V_MODEL_NAMES, BERT_MODELS_NAMES
from src.core.ml.nlp_features_utils import NlpFeaturesUtils
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import csv
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
import numpy as np

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
        train_classification_file_report = open(os.path.join(save_folder, 'train_report.txt'), 'w+', newline='')
        train_classification_file_report.write(train_classification_report)
        train_classification_file_report.close()


        test_preds = model.predict(X_test)
        test_classification_report = classification_report(y_test, test_preds)
        test_classification_report_file = open(os.path.join(save_folder, 'test_report.txt'), 'w+',  newline='')
        test_classification_report_file.write(test_classification_report)
        test_classification_report_file.close()
  
        with open(os.path.join(save_folder, 'predictions.csv'), 'w+', newline='') as predictions_file : 
            writer = csv.writer(predictions_file)
            writer.writerow(['_id', 'label', 'prediction'])

            for pred_idx, prediction in enumerate(test_preds) : 
                writer.writerow([test_df.iloc[pred_idx][MONGO_DB_CONSTANTS.ID_FIELD],y_test[pred_idx], prediction])

        predictions_file.close()

        test_accuracy = accuracy_score(test_preds, y_test)
        return test_accuracy

    @staticmethod
    def use_trained_model_to_predict_tone(model_path, input_text, feature_vector_format, idf_data_path, vocabulary_data_path, w2v_converters, bert_models_tokenizers) :
        """
        Uses a trained machine learning model to predict the tone of the given input text.

        Parameters:
        - model_path: The path to the trained machine learning model file.
        - input_text: The input text for which to predict the tone.
        - feature_vector_format: The format of the feature vectors.
        - idf_data_path: The path to the IDF data file.
        - vocabulary_data_path : The path to the vocablary data file.
        - w2v_converters (list): List of Word2Vec converters.
        - bert_models_tokenizers (list): A list containing tuples of pre-trained BERT or RoBERTa models along with their tokenizers.

        Returns:
        - prediction: The predicted tone of the input text.
        """

        model = joblib.load(model_path)

        encoder = LabelBinarizer() if isinstance(model, MLPClassifier) or isinstance(model, nn.Module) else LabelEncoder()
        
        encoder.fit(TONES_CONSTANTS.TONES)

        feature_vector = None


        if feature_vector_format == FEATURE_FORMAT_CONSTANTS.BOW or feature_vector_format == FEATURE_FORMAT_CONSTANTS.TF_IDF :
            
            bow_feature_vector, tf_idf_feature_vector = NlpFeaturesUtils.generate_bow_tf_idf_feature_vectors(input_text, None, None, idf_data_path, vocabulary_data_path)

            if feature_vector_format == FEATURE_FORMAT_CONSTANTS.BOW : 
                feature_vector = bow_feature_vector
            else : 
                feature_vector = tf_idf_feature_vector
        
        elif 'w2v' in feature_vector_format:
            feature_vectors_by_model =  NlpFeaturesUtils.generate_word2vec_feature_vectors(input_text, w2v_converters)
            for (converted_idx, (max_dims, sum_dims, mean_dims)) in enumerate(feature_vectors_by_model): 
                if W2V_MODEL_NAMES[converted_idx] in feature_vector_format :
                   
                    if 'max' in feature_vector_format:
                        feature_vector =  max_dims
                    
                    elif 'sum' in feature_vector_format:
                        feature_vector = sum_dims
                    
                    else : 
                        feature_vector = mean_dims
                    
                    break
        
        else :
            bert_feature_vectors = NlpFeaturesUtils.generate_bert_feature_vectors(input_text,bert_models_tokenizers)

            for model_idx, model_name in enumerate(BERT_MODELS_NAMES) :
                if model_name == feature_vector_format :
                    feature_vector = bert_feature_vectors[model_idx]
                    break

        
        df = pd.DataFrame(np.array(feature_vector).reshape(1,-1), columns=[idx for idx in range(len(feature_vector))])
        prediction = model.predict(df)
        predicted_tone = encoder.inverse_transform(prediction)
        return predicted_tone[0]









    

