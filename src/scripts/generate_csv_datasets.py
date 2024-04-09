import csv
import json
import os
from src.core.ml_utils.ml_utils import MlUtils
import sys
from src.core.read_env import get_env_variables
from src.core.constants import ENV_CONSTANTS, MONGO_DB_CONSTANTS
from src.core.data_generation.mongo_client import Mongo_Client
import re
import math
from gensim.models import Word2Vec


def get_csv_columns_name(train_voc_set) : 
    columns_name = [MONGO_DB_CONSTANTS.ID_FIELD, MONGO_DB_CONSTANTS.TONE_FIELD] + [idx for idx, _ in enumerate(list(train_voc_set))]
    return columns_name

def get_columns_name_word2_vec() :
    columns_name = [MONGO_DB_CONSTANTS.ID_FIELD, MONGO_DB_CONSTANTS.TONE_FIELD] + [idx for idx in range(300)]
    return columns_name

def generate_word2vec_feature_vector(word2vec_converter, words) :
    """
    Generates feature vectors from Word2Vec embeddings for a given list of words (text).
    Computes three types of feature vectors: maximum, sum, and mean.

    Parameters:
    - word2vec_converter: Word2Vec converter object that maps words to their corresponding embeddings.
    - words: Text in the form of a list of words for which feature vectors are to be generated.

    Returns:
    - Tuple containing three types of feature vectors: max_dims, sum_dims, and mean_dims.
      - max_dims: Feature vector representing the maximum value for each dimension across all word embeddings.
      - sum_dims: Feature vector representing the sum of values for each dimension across all word embeddings.
      - mean_dims: Feature vector representing the mean value for each dimension across all word embeddings.
    """
    max_dims = [- math.inf ] * 300
    sum_dims = [0] * 300
    for word in words : 
        vector = word2vec_converter[word]
        for dim_idx, dim_value in enumerate(vector) : 
            max_dims[dim_idx] = max(max_dims[dim_idx], dim_value)
            sum_dims[dim_idx] += dim_value
    
    mean_dims = [dim_sum / len(words) for dim_sum in sum_dims]

    return max_dims, sum_dims, mean_dims



def create_multi_representation_datasets_csv(data, train_voc_set, words_idf_data, w2v_converter, output_folder, file_name) : 
    """
    Generates split datasets for Bag-of-Words (BoW), TF-IDF, and 3 Word2Vec representations (max, mean, sum).

    Parameters:
    - data: List of data entries.
    - train_voc_set: Set of vocabulary words used for training.
    - words_idf_data: Dictionary containing IDF values for words.
    - w2v_converter: Word2Vec converter object.
    - output_folder: Path to the output folder for storing the datasets.
    - file_name: Name of the output files (train or test).

    """

    # Open files for writing
    bow_file  = open(os.path.join(output_folder, f'bow_{file_name}csv'), 'w+')
    tf_idf_file = open(os.path.join(output_folder, f'tf_idf_{file_name}.csv'), 'w+')
    w2v_max_file = open(os.path.join(output_folder, f'w2v_max_{file_name}.csv'), 'w+')
    w2v_sum_file = open(os.path.join(output_folder, f'w2v_sum_{file_name}.csv'), 'w+')
    w2v_mean_file = open(os.path.join(output_folder, f'w2v_mean_{file_name}.csv'), 'w+')

    # Create CSV writers
    bow_writer =  csv.writer(bow_file)
    tf_idf_writer = csv.writer(tf_idf_file)
    w2v_max_writer = csv.writer(w2v_max_file)
    w2v_sum_writer = csv.writer(w2v_sum_file)
    w2v_mean_writer = csv.writer(w2v_mean_file)
    
    # Write column headers
    bow_writer.writerow(get_csv_columns_name(train_voc_set))
    tf_idf_writer.writerow(get_csv_columns_name(train_voc_set))
    w2v_max_writer.writerow(get_columns_name_word2_vec())
    w2v_sum_writer.writerow(get_columns_name_word2_vec())
    w2v_mean_writer.writerow(get_columns_name_word2_vec())

    for entry in data :
        new_row = [str(entry[MONGO_DB_CONSTANTS.ID_FIELD]), entry[MONGO_DB_CONSTANTS.TONE_FIELD]]
        words = re.split(r'\W+', entry[MONGO_DB_CONSTANTS.TEXT_FIELD])
        text_words_count = {word : 0 for word in train_voc_set}

        n_terms = len(words)

        # Compute word counts for words in the vocabulary set
        for word in words :
            if word not in train_voc_set :
                continue
            text_words_count[word] += 1

        
        bow_writer.writerow(new_row + [text_words_count[word] for word in list(train_voc_set)])
        tf_idf_writer.writerow(new_row + [(text_words_count[word]/ n_terms) * words_idf_data[word] for word in list(train_voc_set)])

        max, sum, mean = generate_word2vec_feature_vector(w2v_converter, words)
        w2v_max_writer.writerow(new_row + max)
        w2v_sum_writer.writerow(new_row + sum)
        w2v_mean_writer.writerow(new_row + mean)
     
    # Close files
    bow_file.close()
    tf_idf_file.close()
    w2v_max_file.close()
    w2v_sum_file.close()
    w2v_mean_file.close()

def generate_multi_representation_split_csv(train_ids, test_ids, split_output_folder, w2v_converter) :
    """
    Generates split CSV files with multiple representations (Bag-of-Words (BoW), TF-IDF, and 3 Word2Vec representations (max, mean, sum)) for training and testing datasets.

    Parameters:
    - train_ids: List of IDs for training data.
    - test_ids: List of IDs for testing data.
    - split_output_folder: Path to the output folder for storing the split CSV files.
    - w2v_converter: Word2Vec converter object.
    """

    train_data = mongo_client.get_collection_data({MONGO_DB_CONSTANTS.ID_FIELD : {'$in' : train_ids}}, {MONGO_DB_CONSTANTS.TEXT_FIELD : 1, MONGO_DB_CONSTANTS.TONE_FIELD : 1, MONGO_DB_CONSTANTS.ID_FIELD : 1})
    words_idf_data = MlUtils.get_word_idf_data(train_data)
    words_idf_json_file = open(os.path.join(split_output_folder, 'tf-idf-data.json'), 'w+')
    json.dump(words_idf_data, words_idf_json_file)
    words_idf_json_file.close()

    train_voc_set = set(words_idf_data.keys())

    train_data = mongo_client.get_collection_data({MONGO_DB_CONSTANTS.ID_FIELD : {'$in' : train_ids}}, {MONGO_DB_CONSTANTS.TEXT_FIELD : 1, MONGO_DB_CONSTANTS.TONE_FIELD : 1, MONGO_DB_CONSTANTS.ID_FIELD : 1})
    create_multi_representation_datasets_csv(train_data, train_voc_set, words_idf_data, w2v_converter, 'train')
    test_data = mongo_client.get_collection_data({MONGO_DB_CONSTANTS.ID_FIELD : {'$in' : test_ids}}, {MONGO_DB_CONSTANTS.TEXT_FIELD : 1, MONGO_DB_CONSTANTS.TONE_FIELD : 1, MONGO_DB_CONSTANTS.ID_FIELD : 1})
    create_multi_representation_datasets_csv(test_data, train_voc_set, words_idf_data, w2v_converter, 'test')

    
if __name__ == '__main__' :
    """
    Generate split datasets with multiple representations for machine learning tasks.
    Retrieves environment variables, connects to MongoDB, loads Word2Vec model, and generates split datasets.
    Generates CSV files containing the split datasets with multiple representations including Bag-of-Words (BoW),
    TF-IDF, and 3 Word2Vec representations (max,mean, sum)
    """

    # Define output folder for storing datasets
    output_folder = "datasets"

    # Retrieve environment variables
    env_variables, error, error_msg = get_env_variables() 
    if error:
        print('An error occurred:')
        print(error_msg)
        sys.exit(1)
    
    try:
        # Initialize MongoDB client
        mongo_client = Mongo_Client(env_variables[ENV_CONSTANTS.MONGO_URI_FIELD])
        mongo_client.connect_to_db(ENV_CONSTANTS.MONGO_DB_NAME_FIELD, ENV_CONSTANTS.DB_CLEAN_COLLECTION_FIELD)
        
        # Load pre-trained Word2Vec model
        word2vec_converter = Word2Vec.load(os.path.join('models', 'word2vec-google-news-300.model'))

        # Get balanced k-fold splits
        splits = MlUtils.get_k_folds_balanced_splits_ids(mongo_client, k=3)

        # Iterate over splits
        for split_idx, split in enumerate(splits):
            train_ids, test_ids = split[0], split[1]
        
            # Create output folder for the split
            split_output_folder = os.path.join(output_folder, f'split{split_idx}')
            os.mkdir(split_output_folder)

            # Generate split datasets with multiple representations
            generate_multi_representation_split_csv(train_ids, test_ids, split_output_folder, word2vec_converter)
    
    except Exception as e:
        print('An error occurred:')
        print(e)
        sys.exit(1)