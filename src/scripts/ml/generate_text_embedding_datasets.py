import csv
import os
from src.core.ml.ml_dataset_utils import MlDatasetUtils
from src.core.ml.nlp_features_utils import NlpFeaturesUtils
import sys
from src.core.read_env import get_env_variables
from src.core.constants import ENV_CONSTANTS, MONGO_DB_CONSTANTS, FEATURE_FORMAT_CONSTANTS
from src.core.text_generation.mongo_client import Mongo_Client
import traceback

import re

def generate_text_embedding_split_csv(data, vocabulary, words_idf_data, output_folder, file_name, w2v_converters, bert_models_tokenizers) : 
    """
    Generates split datasets for  Bag-of-Words (BoW), TF-IDF, 3 Word2Vec text embedding methods (max, mean and sum pooling) 
    with 5 different Word2Vec models for words embedding, and 3 BERT models for text embedding.

    Parameters:
    - data: List of data entries.
    - vocabulary: List of vocabulary words used for training.
    - words_idf_data: Dictionary containing IDF values for words.
    - output_folder: Path to the output folder for storing the datasets.
    - file_name: Name of the output files (train or test).
    - w2v_converters (list): List of loaded Word2Vec models.
    - bert_models_tokenizers (list): A list containing tuples of pre-trained BERT or RoBERTa models along with their tokenizers.

    """
    # Open files for writing
    files = [open(os.path.join(output_folder, f'{feature_name}_{file_name}.csv'), 'w+', newline='') for feature_name in FEATURE_FORMAT_CONSTANTS.FEATURES_NAMES]
    writers = [csv.writer(file) for file in files]
    
    # Write column headers
    columns_name = [MONGO_DB_CONSTANTS.ID_FIELD, MONGO_DB_CONSTANTS.TONE_FIELD] 
    for writer_idx, writer in enumerate(writers) :
        corresponding_feature_name = FEATURE_FORMAT_CONSTANTS.FEATURES_NAMES[writer_idx] 
        if corresponding_feature_name == FEATURE_FORMAT_CONSTANTS.BOW or corresponding_feature_name == FEATURE_FORMAT_CONSTANTS.TF_IDF :
            writer.writerow(columns_name + [idx for idx in range(len(vocabulary))])
            continue
        
        elif 'w2v' in corresponding_feature_name : 

            n_w2v_dim = int(re.search(r'\d+', corresponding_feature_name[3:]).group())
            writer.writerow(columns_name + [idx for idx in range(n_w2v_dim)])
        
        else : 
            feature_size = NlpFeaturesUtils.get_bert_feature_vector_size(corresponding_feature_name)
            writer.writerow(columns_name + [idx for idx in range(feature_size)])


    for entry in data :
        new_row = [str(entry[MONGO_DB_CONSTANTS.ID_FIELD]), entry[MONGO_DB_CONSTANTS.TONE_FIELD]]
        bow_feature_vector, tf_idf_feature_vector = NlpFeaturesUtils.generate_bow_tf_idf_feature_vectors(entry[MONGO_DB_CONSTANTS.TEXT_FIELD], words_idf_data, vocabulary)

        writers[0].writerow(new_row + bow_feature_vector)
        writers[1].writerow(new_row + tf_idf_feature_vector)
        
        
        w2v_features_by_models = NlpFeaturesUtils.generate_word2vec_feature_vectors(entry[MONGO_DB_CONSTANTS.TEXT_FIELD], w2v_converters)

        writer_idx = 2
        for max,sum, mean in w2v_features_by_models : 
                writers[writer_idx].writerow(new_row + max)
                writers[writer_idx + 1].writerow(new_row + sum)
                writers[writer_idx + 2].writerow(new_row + mean)
                writer_idx += 3
        
        bert_features_by_models = NlpFeaturesUtils.generate_bert_feature_vectors(entry[MONGO_DB_CONSTANTS.TEXT_FIELD], bert_models_tokenizers)
        for idx in range(3) :
            writers[writer_idx].writerow(new_row + bert_features_by_models[idx])
            writer_idx += 1
     
    # Close files
    [file.close() for file in files]

def generate_text_embedding_split_csv_wrapper(train_ids, test_ids, split_output_folder) :
    """
    Generates split CSV files with multiple text embeddings : Bag-of-Words (BoW), TF-IDF, 3 Word2Vec text embedding methode (max, mean and sum pooling) with 5 different Word2Vec models for words embedding, and 3 BERT models for text embedding.
      (Bag-of-Words (BoW), TF-IDF, 3 Word2Vec text representations (max, mean, sum) with 5 different Word2Vec models) and bert texts representation for training and testing datasets.
    It also downloads word2vec and Bert converters if they don't exist.
    Parameters:
    - train_ids: List of IDs for training data.
    - test_ids: List of IDs for testing data.
    - split_output_folder: Path to the output folder for storing the split CSV files.
    """

    w2v_converters = NlpFeaturesUtils.load_word2vec_converters()
    bert_models_tokenizers = NlpFeaturesUtils.load_bert_converters()

    train_data = mongo_client.get_collection_data({MONGO_DB_CONSTANTS.ID_FIELD : {'$in' : train_ids}}, {MONGO_DB_CONSTANTS.TEXT_FIELD : 1, MONGO_DB_CONSTANTS.TONE_FIELD : 1, MONGO_DB_CONSTANTS.ID_FIELD : 1})
    words_idf_data, vocabulary = NlpFeaturesUtils.get_words_idf_and_voc(train_data, split_output_folder)

    train_data = mongo_client.get_collection_data({MONGO_DB_CONSTANTS.ID_FIELD : {'$in' : train_ids}}, {MONGO_DB_CONSTANTS.TEXT_FIELD : 1, MONGO_DB_CONSTANTS.TONE_FIELD : 1, MONGO_DB_CONSTANTS.ID_FIELD : 1})
    generate_text_embedding_split_csv(train_data, vocabulary, words_idf_data, split_output_folder, 'train', w2v_converters, bert_models_tokenizers)
    test_data = mongo_client.get_collection_data({MONGO_DB_CONSTANTS.ID_FIELD : {'$in' : test_ids}}, {MONGO_DB_CONSTANTS.TEXT_FIELD : 1, MONGO_DB_CONSTANTS.TONE_FIELD : 1, MONGO_DB_CONSTANTS.ID_FIELD : 1})
    generate_text_embedding_split_csv(test_data, vocabulary, words_idf_data, split_output_folder, 'test', w2v_converters, bert_models_tokenizers)

if __name__ == '__main__' :
    """
    Generate split datasets with multiple text embeddings for machine learning tasks.
    
    Retrieves environment variables, connects to MongoDB, and generates split datasets.
    Generates CSV files containing the split datasets with multiple text embeddings : Bag-of-Words (BoW), TF-IDF, 3 Word2Vec text embedding methode (max, mean and sum pooling) 
    with 5 different Word2Vec models for words embedding, and 3 BERT models for text embedding.

     Parameters:
    - output_folder (str): The folder where the trained datasets will be saved.
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
        mongo_client.connect_to_db(env_variables[ENV_CONSTANTS.MONGO_DB_NAME_FIELD], env_variables[ENV_CONSTANTS.DB_CLEAN_COLLECTION_FIELD])
      
        # Get balanced k-fold splits
        splits = MlDatasetUtils.get_k_folds_balanced_splits_ids(mongo_client, k=3)
        
        
        # Iterate over splits
        for split_idx, split in enumerate(splits):
            train_ids, test_ids = split[0], split[1]
        
            # Create output folder for the split
            split_output_folder = os.path.join(output_folder, f'split{split_idx}')
            os.mkdir(split_output_folder)


            # Generate split datasets with multiple representations
            generate_text_embedding_split_csv_wrapper(train_ids, test_ids, split_output_folder)
    
    except Exception as e:
        print('An error occurred:')
        print(e)
        traceback_info = traceback.format_exc()
        print(traceback_info)
        sys.exit(1)