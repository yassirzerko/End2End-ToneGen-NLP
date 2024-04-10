import random
import math
from src.core.constants import TONES_CONSTANTS, SIZE_CONSTANTS, MONGO_DB_CONSTANTS, FEATURE_FORMAT_CONSTANTS
import re
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os
import json
from gensim.models import Word2Vec
import csv
from itertools import product

class MlUtils : 
    @staticmethod
    def get_k_folds_balanced_splits_ids(db_client, k = 5) :
        """
        Generates balanced k-fold splits of data IDs for cross-validation.

        Parameters:
        - db_client: Database client object.
        - k: Number of folds for cross-validation (default is 5).

        Returns:
        - List of k splits, where each split is a list containing two lists: train_ids and test_ids.
        """
        folds_ids = MlUtils.get_balanced_folds_ids(db_client, k)
        splits = [ [[], []] for _ in range(k)]

        for n_fold in range(k) :
            splits[n_fold][1] += folds_ids[n_fold]

            for n_fold_s in range(k) :
                if n_fold_s == n_fold : 
                    continue
                
                splits[n_fold][0] += folds_ids[n_fold_s]
        
        return splits

    @staticmethod
    def get_balanced_folds_ids(db_client, k = 5) :
        """
        Retrieves balanced folds of data IDs from a database client.

        Parameters:
        - db_client: Database client object.
        - k: Number of folds for cross-validation (default is 5).

        Returns:
        - List of k lists, where each list contains data IDs for a fold.

        Note:
        The folds are balanced by label (Tone) and by text size.
        """

        folds_ids = [ [] for _ in range (k) ]
        for tone in TONES_CONSTANTS.TONES :
            for size in [SIZE_CONSTANTS.SMALL, SIZE_CONSTANTS.MEDIUM, SIZE_CONSTANTS.LARGE, SIZE_CONSTANTS.VERY_LARGE] :
                data = list(db_client.get_collection_data({MONGO_DB_CONSTANTS.TONE_FIELD : tone, MONGO_DB_CONSTANTS.LABEL_SIZE_FIELD : size}, {MONGO_DB_CONSTANTS.ID_FIELD : 1}))
                random.shuffle(data)
                nb_entries_by_fold = int(len(data) / k)

                for n_fold in range (k) : 
                    start_range = n_fold*nb_entries_by_fold 
                    end_range = start_range + nb_entries_by_fold
                    if end_range > len(data) :
                        end_range = len(data)
                    
                    if n_fold == k - 1 and end_range < len(data) :
                        end_range = len(data)
                    fold_data = data[start_range : end_range]                  

                    folds_ids[n_fold] += [entry[MONGO_DB_CONSTANTS.ID_FIELD] for entry in fold_data]

            
        return folds_ids
    

    @staticmethod
    def get_word_idf_data(data) :
        """
        Computes the inverse document frequency (IDF) of words in a given dataset.

        Parameters:
        - data: List of data entries.

        Returns:
        - Dictionary containing IDF values for each unique word in the dataset.
        """
        
        words_idf = {}
        n_texts = 0

        for entry in data :
            words = re.split(r'\W+', entry['Text'])
            uniques = list(set(words))
            n_texts  += 1
            for word in uniques :
                if word =='' :
                    continue
                if word not in words_idf :
                    words_idf[word] = 0
                words_idf[word] += 1

        words_idf = {word : math.log(n_texts) / (1 + words_idf[word]) for word in words_idf}
        return words_idf
    
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
    def get_words_count_data(text, train_voc_set) :
        """
        Computes the count of each word in the vocabulary set within the given text 
        and the total number of terms (non-empty words) in the text.

        Parameters:
        - text: The input text to analyze.
        - train_voc_set: A set containing the vocabulary words to count occurrences for.

        Returns:
        - text_words_count: A dictionary containing the count of each word from the vocabulary set in the text.
        - n_terms: The total number of terms (non-empty words) in the text.
        """
        words = re.split(r'\W+', text)
        text_words_count = {word : 0 for word in train_voc_set}

        n_terms = 0

        # Compute word counts for words in the vocabulary set
        for word in words :
            if word == '' :
                continue 
            n_terms += 1

            if word not in train_voc_set : 
                continue
            text_words_count[word] += 1
        
        return text_words_count, n_terms
    
    @staticmethod
    def generate_bow_tf_idf_feature_vector(idf_data,text) :
        """
        Generates Bag-of-Words (BoW) and TF-IDF feature vectors for the given text.

        Parameters:
        - idf_data: A dictionary containing inverse document frequency (IDF) scores for words in the vocabulary.
        - text: The input text for which to generate the feature vectors.

        Returns:
        - bow_feature_vector: Bag-of-Words (BoW) feature vector for the text.
        - tf_idf_feature_vector: TF-IDF feature vector for the text.
        """
        train_voc_set = set(idf_data.keys())
        text_words_count, n_terms = MlUtils.get_words_count_data(text, train_voc_set)

        bow_feature_vector = [text_words_count[word] for word in list(train_voc_set)]
        tf_idf_feature_vector = [(text_words_count[word] / n_terms) * idf_data[word] for word in list(train_voc_set)]

        return bow_feature_vector, tf_idf_feature_vector
    
    @staticmethod
    def generate_word2vec_feature_vector(word2vec_converter, text) :
        """
        Generates feature vectors from Word2Vec embeddings for a given list of words (text).
        Computes three types of feature vectors: maximum, sum, and mean.

        Parameters:
        - word2vec_converter: Word2Vec converter object that maps words to their corresponding embeddings.
        - text: Text for which feature vectors are to be generated.

        Returns:
        - Tuple containing three types of feature vectors: max_dims, sum_dims, and mean_dims.
        - max_dims: Feature vector representing the maximum value for each dimension across all word embeddings.
        - sum_dims: Feature vector representing the sum of values for each dimension across all word embeddings.
        - mean_dims: Feature vector representing the mean value for each dimension across all word embeddings.
        """
        words = re.split(r'\W+', text)

        max_dims = [- math.inf ] * 300
        sum_dims = [0] * 300
        word_counter = 0
        for word in words : 
            if word == '' : 
                continue
            word_counter += 1
            if word not in word2vec_converter :
                continue
            vector = word2vec_converter[word]
            for dim_idx, dim_value in enumerate(vector) : 
                max_dims[dim_idx] = max(max_dims[dim_idx], dim_value)
                sum_dims[dim_idx] += dim_value
        
        mean_dims = [dim_sum / word_counter for dim_sum in sum_dims]

        return max_dims, sum_dims, mean_dims
    
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
            if not 'tf-idf-data-path'in kwargs : 
                print('Error')
                return
            tf_idf_data = json.load(kwargs['tf-idf-data-path'])
            bow_feature_vector, tf_idf_feature_vector = MlUtils.generate_bow_tf_idf_feature_vector(tf_idf_data, input_text)

            if feature_vector_format == FEATURE_FORMAT_CONSTANTS.BOW:
                prediction = model.predict(bow_feature_vector)
                return prediction
            return model.predict(tf_idf_feature_vector)

        if not 'w2v_model_path' in kwargs : 
            print('Error')
            return 
        
        w2v_converter = Word2Vec.load(kwargs['w2v_model_path'])
        
        max_dims, sum_dims, mean_dims = MlUtils.generate_word2vec_feature_vector(w2v_converter, input_text)
        if feature_vector_format == FEATURE_FORMAT_CONSTANTS.W2V_MAX:
            return model.predict(max_dims)
        if feature_vector_format == FEATURE_FORMAT_CONSTANTS.W2V_SUM :
            return sum_dims
        return mean_dims









    

