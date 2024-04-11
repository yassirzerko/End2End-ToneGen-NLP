import re
import math 
from gensim.models import Word2Vec
import json
import os

class NlpFeaturesUtils : 
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
    def get_word_idf_data(data, save_foler = None) :
        """
        Computes the inverse document frequency (IDF) of words in a given dataset.

        Parameters:
        - data: List of data entries.
        - save_folder: Optional. Folder path where the IDF data JSON file will be saved.

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
        if save_foler != None : 
            idf_json_file = open(os.path.join(save_foler, 'tf-idf-data.json'), 'w+')
            json.dump(words_idf, idf_json_file)
            idf_json_file.close()

        return words_idf
    
    @staticmethod
    def generate_bow_tf_idf_feature_vector(text, idf_data = None, idf_data_file = None) :
        """
        Generates Bag-of-Words (BoW) and TF-IDF feature vectors for the given text.

        Parameters:
        - idf_data: A dictionary containing inverse document frequency (IDF) scores for words in the vocabulary. 
                    Default is None.
        - idf_data_file: Optional. The path to a JSON file containing IDF data. Default is None.

        Returns:
        - bow_feature_vector: Bag-of-Words (BoW) feature vector for the text.
        - tf_idf_feature_vector: TF-IDF feature vector for the text.

        Raises:
        - Exception: If both idf_data and idf_data_file are None, indicating missing IDF data.
        """

        if idf_data == None  : 
            if idf_data_file == None :
                raise Exception('Missing idf data, either provide a loaded idf dict or a json to an idf dict.')

            idf_data = json.load(idf_data_file) 

        train_voc_set = set(idf_data.keys())
        text_words_count, n_terms = NlpFeaturesUtils.get_words_count_data(text, train_voc_set)

        bow_feature_vector = [text_words_count[word] for word in list(train_voc_set)]
        tf_idf_feature_vector = [(text_words_count[word] / n_terms) * idf_data[word] for word in list(train_voc_set)]

        return bow_feature_vector, tf_idf_feature_vector
    
    @staticmethod
    def generate_word2vec_feature_vector(word2vec_converter_path, text) :
        """
        Generates feature vectors from Word2Vec embeddings for a given list of words (text).
        Computes three types of feature vectors: maximum, sum, and mean.

        Parameters:
        - word2vec_converter_path: Word2Vec converter object path.
        - text: Text for which feature vectors are to be generated.

        Returns:
        - Tuple containing three types of feature vectors: max_dims, sum_dims, and mean_dims.
        - max_dims: Feature vector representing the maximum value for each dimension across all word embeddings.
        - sum_dims: Feature vector representing the sum of values for each dimension across all word embeddings.
        - mean_dims: Feature vector representing the mean value for each dimension across all word embeddings.
        """
        word2vec_converter = Word2Vec.load(word2vec_converter_path, mmap='r')
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