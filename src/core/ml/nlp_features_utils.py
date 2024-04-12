import re
import math 
from gensim.models import KeyedVectors
import json
import os
from src.core.constants import W2V_MODEL_NAMES

class NlpFeaturesUtils : 
    @staticmethod
    def get_words_count_data(text, vocabulary) :
        """
        Computes the count of each word in the vocabulary set within the given text 
        and the total number of terms (non-empty words) in the text.

        Parameters:
        - text: The input text to analyze.
        - vocabulary: A list containing the vocabulary.

        Returns:
        - text_words_count: A dictionary containing the count of each word from the vocabulary set in the text.
        - n_terms: The total number of terms (non-empty words) in the text.
        """
        words = re.split(r'\W+', text)
        text_words_count = {word : 0 for word in vocabulary}

        n_terms = 0

        # Compute word counts for words in the vocabulary set
        for word in words :
            if word == '' :
                continue 
            n_terms += 1

            if word not in vocabulary : 
                continue
            text_words_count[word] += 1
        
        return text_words_count, n_terms

    @staticmethod
    def get_words_idf_and_voc(data, save_foler = None) :
        """
        Computes the inverse document frequency (IDF) of words in a given dataset and the vocabulary.

        Parameters:
        - data: List of data entries.
        - save_folder: Optional. Folder path where the IDF data JSON file will be saved.

        Returns:
        - Dictionary containing IDF values for each unique word in the dataset.
        - List containing the vocabulary.
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
        vocabulary = list(words_idf.keys())
        if save_foler != None : 
            idf_json_file = open(os.path.join(save_foler, 'tf-idf-data.json'), 'w+')
            voc_file = open(os.path.join(save_foler, 'voc.json'), 'w+')
            json.dump(words_idf, idf_json_file)
            json.dump(vocabulary, voc_file)
            idf_json_file.close()
            voc_file.close()

        return words_idf, vocabulary
    
    @staticmethod
    def generate_bow_tf_idf_feature_vectors(text, idf_data = None, vocabulary = None, idf_data_file_path = None, vocabulary_file_path = None) :
        """
        Generates Bag-of-Words (BoW) and TF-IDF feature vectors for the given text.

        Parameters:
        - idf_data: A dictionary containing inverse document frequency (IDF) scores for words in the vocabulary. 
        - idf_data_file_path: The path to a JSON file containing IDF data.
        - vocabulary: A list containing the vocabulary
        - vocaulary_file_path: The path to a JSON file containing the vocabulary.

        Returns:
        - bow_feature_vector: Bag-of-Words (BoW) feature vector for the text.
        - tf_idf_feature_vector: TF-IDF feature vector for the text.

        Raises:
        - Exception: Should provide either the loaded data or the JSON file path for idf_data and the vocabulary
        """

        if idf_data == None  : 
            if idf_data_file_path == None :
                raise Exception('Missing idf data, either provide a loaded idf dict or a json to an idf dict.')

            file = open(idf_data_file_path, 'r')
            idf_data = json.load(file) 
            file.close()
        
        if vocabulary == None  : 
            if vocabulary_file_path == None :
                raise Exception('Missing vocabulary data, either provide a loaded vocabulary list or a json.')

            file = open(vocabulary_file_path, 'r')
            vocabulary  = json.load(file) 
            file.close()
            
        text_words_count, n_terms = NlpFeaturesUtils.get_words_count_data(text, vocabulary)

        bow_feature_vector = [text_words_count[word] for word in vocabulary]
        tf_idf_feature_vector = [(text_words_count[word] / n_terms) * idf_data[word] for word in vocabulary]

        return bow_feature_vector, tf_idf_feature_vector
    
    @staticmethod
    def generate_word2vec_feature_vectors(text) :
        """
        Generates feature vectors from Word2Vec embeddings for a given text.
        Computes three types of feature vectors: maximum, sum, and mean, across multiple Word2Vec models.

        Parameters:
        - text: Text for which feature vectors are to be generated.

        Returns:
        - List of tuples containing three types of feature vectors for each Word2Vec model:
            - max_dims: Feature vector representing the maximum value for each dimension across all word embeddings.
            - sum_dims: Feature vector representing the sum of values for each dimension across all word embeddings.
            - mean_dims: Feature vector representing the mean value for each dimension across all word embeddings.
        """

        converters_data = []
        for word2vec_converter_path  in W2V_MODEL_NAMES :

            word2vec_converter = KeyedVectors.load(word2vec_converter_path, mmap='r')
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

            converters_data.append([max_dims, sum_dims, mean_dims])

        return converters_data