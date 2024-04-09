import random
import math
from src.core.constants import TONES_CONSTANTS, SIZE_CONSTANTS, MONGO_DB_CONSTANTS, FEATURE_FORMAT_CONSTANTS
import re
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import joblib
import os
import json
from gensim.models import Word2Vec
import csv

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
                    nb_ids_to_add = nb_entries_by_fold if  k*nb_entries_by_fold + nb_entries_by_fold <= len(data) else len(data)
                    fold_data = data[k*nb_entries_by_fold : k*nb_entries_by_fold + nb_ids_to_add]

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
                if word not in words_idf :
                    words_idf[word] = 0
                words_idf[word] += 1

        words_idf = {word : math.log(n_texts) / (1 + words_idf[word]) for word in words_idf}
        return words_idf
    
    @staticmethod
    def execute_train_test(train_path, test_path, model, save_folder, model_name) :
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train, y_train = train_df.drop(columns=[MONGO_DB_CONSTANTS.GENERATED_BY_FIELD, MONGO_DB_CONSTANTS.TONE_FIELD,  MONGO_DB_CONSTANTS.ID_FIELD]), train_df[MONGO_DB_CONSTANTS.TONE_FIELD]
        X_test, y_test = test_df.drop(columns=[MONGO_DB_CONSTANTS.GENERATED_BY_FIELD, MONGO_DB_CONSTANTS.TONE_FIELD,  MONGO_DB_CONSTANTS.ID_FIELD]), test_df[MONGO_DB_CONSTANTS.TONE_FIELD]

        one_hot_encoder = LabelBinarizer()
        y_train = one_hot_encoder.fit_transform(y_train)
        y_test = one_hot_encoder.transform(y_test)


        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)

        print(classification_report(y_test, test_preds))
        joblib.dump(model, os.path.join(save_folder, model_name + '.pkl'))
        with open(os.path.join(save_folder, 'predictions.csv')) as predictions_file : 
            writer = csv.writer(predictions_file)
            writer.writerow(['_id', 'label', 'prediction'])

            for pred_idx, prediction in enumerate(test_preds) : 
                writer.writerow([test_df.iloc[pred_idx][MONGO_DB_CONSTANTS.ID_FIELD],y_test[pred_idx], prediction])

            writer.close()


    @staticmethod
    def get_words_count_data(text, train_voc_set) :
        words = re.split(r'\W+', text)
        text_words_count = {word : 0 for word in train_voc_set}

        n_terms = len(words)

        # Compute word counts for words in the vocabulary set
        for word in words :
            if word not in train_voc_set :
                continue
            text_words_count[word] += 1
        
        return text_words_count, n_terms
    
    @staticmethod
    def generate_bow_tf_idf_feature_vector(tf_idf_data,text) :
        train_voc_set = set(tf_idf_data.keys())
        text_words_count, n_terms = MlUtils.get_words_count_data(text, train_voc_set)

        bow_feature_vector = [text_words_count[word] for word in list(train_voc_set)]
        tf_idf_feature_vector = [(text_words_count[word]/ n_terms) * tf_idf_data[word] for word in list(train_voc_set)]

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
        for word in words : 
            vector = word2vec_converter[word]
            for dim_idx, dim_value in enumerate(vector) : 
                max_dims[dim_idx] = max(max_dims[dim_idx], dim_value)
                sum_dims[dim_idx] += dim_value
        
        mean_dims = [dim_sum / len(words) for dim_sum in sum_dims]

        return max_dims, sum_dims, mean_dims
    
    @staticmethod
    def use_trained_model_to_preidct_tone(model_path, input_text, **kwargs) :
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









    

