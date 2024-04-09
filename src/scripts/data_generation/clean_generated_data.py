
import re
from src.core.constants import TONES_CONSTANTS, OpenAIConstants, SIZE_CONSTANTS, ENV_CONSTANTS, MONGO_DB_CONSTANTS
from src.core.data_generation.gpt_text_generator import GPT_Text_Generator
from src.core.data_generation.mongo_client import Mongo_Client
from src.core.read_env import get_env_variables
import sys

def clean_number_pattern(collection, entry) :
    """
    Cleans number patterns from text entries in a MongoDB collection.

    Parameters:
    - collection: The MongoDB collection containing the entries.
    - entry: The entry to be cleaned.

    Description:
    This function removes number patterns from text entries in a MongoDB collection.
    If there's only one sentence found after splitting the text, it updates the existing entry in the collection.
    If there are multiple sentences, it updates the existing entry with the first sentence and inserts new entries for the rest,
    each with the same metadata as the original entry but with the text replaced by a single sentence.
    The "generated_by" field of the new entries is modified to include '_cleaning' to indicate the cleaning process.
    """
    id_to_change = entry[MONGO_DB_CONSTANTS.ID_FIELD]
    text = entry[MONGO_DB_CONSTANTS.TEXT_FIELD]
    generated_by_cleaning = entry[MONGO_DB_CONSTANTS.GENERATED_BY_FIELD] + '_cleaning'

    split_pattern = r'\b\d+\.\s*'
    sentences = re.split(split_pattern, text)
    sentences = sentences[1:]

    if len(sentences) == 1 :
        collection.update_one({MONGO_DB_CONSTANTS.ID_FIELD: id_to_change}, {'$set' : {MONGO_DB_CONSTANTS.TEXT_FIELD: sentences[0], MONGO_DB_CONSTANTS.GENERATED_BY_FIELD : generated_by_cleaning}})
        return

    saved_sentences = 0
    for sentence in sentences :
        if sentence.strip() == '' :
            continue

        if saved_sentences == 0 :
            collection.update_one({MONGO_DB_CONSTANTS.ID_FIELD : id_to_change}, {'$set' : {MONGO_DB_CONSTANTS.TEXT_FIELD : sentence, MONGO_DB_CONSTANTS.GENERATED_BY_FIELD : generated_by_cleaning}})
            saved_sentences += 1
            continue

        # New entry in db
        entry.pop(MONGO_DB_CONSTANTS.ID_FIELD)
        entry[MONGO_DB_CONSTANTS.TEXT_FIELD] = sentence
        entry[MONGO_DB_CONSTANTS.GENERATED_BY_FIELD] = generated_by_cleaning
        collection.insert_one(entry)


def clean_word_counter(collection, entry) :
    """
    Counts the number of words in a text entry and updates the corresponding field in a MongoDB collection.

    Parameters:
    - collection: The MongoDB collection containing the entries.
    - entry: The entry to be processed.
    """
    id_to_change = entry[MONGO_DB_CONSTANTS.ID_FIELD]
    text = entry[MONGO_DB_CONSTANTS.TEXT_FIELD]
    words = re.findall(r'\b\w+\b', text)
    n_words = len(words)
    collection.update_one({MONGO_DB_CONSTANTS.ID_FIELD: id_to_change}, {'$set' : {MONGO_DB_CONSTANTS.N_WORDS_FIELD: n_words}})

def remove_duplicates_texts(collection, entry) :
    """
    Removes duplicate texts in a MongoDB collection except for the first occurrence.

    Parameters:
    - collection: The MongoDB collection containing the entries.
    - entry: The entry containing information about the duplicates.

    Description:
    This function removes duplicate entries in a MongoDB collection except for the first occurrence.
    It uses the '_id' field to identify the text of the entry and 'first_document_id' to identify the first occurrence.
    The function deletes all entries with the same text as the given entry except for the one with the ID specified as 'first_document_id'.
    """

    text = entry['_id']
    first_id = entry['first_document_id']
    collection.delete_many({MONGO_DB_CONSTANTS.TEXT_FIELD : text , MONGO_DB_CONSTANTS.ID_FIELD : {'$ne' : first_id}})

if __name__ == "__main__" :
    """
   Clean and preprocess text data stored in a MongoDB collection.
    
    Retrieves environment variables, initializes MongoDB clients, and performs data cleaning and preprocessing operations.
    The operations include:
    1. Transferring data from the raw collection to the clean collection.
    2. Splitting texts containing numbered patterns into separate entries.
    3. Updating the word count for each text entry.
    4. Adding labeled sizes based on word count thresholds.
    5. Removing duplicate text entries except for the first occurrence.
    """
    
    # Retrieve environment variables
    env_variables, error, error_msg = get_env_variables() 
    if error : 
      print('An error occured : ')
      print(error_msg)
      sys.exit(1)
    
    
    try :      
        # Initialize MongoDB client for the raw collection
        mongo_raw_client = Mongo_Client(env_variables[ENV_CONSTANTS.MONGO_URI_FIELD])
        mongo_raw_client.connect_to_db(ENV_CONSTANTS.MONGO_DB_NAME_FIELD, ENV_CONSTANTS.DB_RAW_COLLECTION_FIELD)

        # Initialize MongoDB client for the clean collection, this collection should be empty before running this script
        mongo_clean_client = Mongo_Client(env_variables[ENV_CONSTANTS.MONGO_URI_FIELD])
        mongo_clean_client.connect_to_db(ENV_CONSTANTS.MONGO_DB_NAME_FIELD, ENV_CONSTANTS.DB_CLEAN_COLLECTION_FIELD)

        # 1 Copy everything to the new database
        mongo_raw_client.transfert_data_to_collection(mongo_clean_client.collection)

        # 2 Split texts that contains many texts separated by '1., 2. ...' into new entries
        contain_number_pattern = re.compile(r'\d+\.')
        number_pattern_query = {'Text' : {'$regex': contain_number_pattern}}
        mongo_clean_client.update_collection_with_fun(number_pattern_query, clean_number_pattern)

        # 3 Update the word counter of all the texts after the changes
        mongo_clean_client.update_collection_with_fun({}, clean_word_counter)

        # 4 Add new labeled size based on word count
        # small < 15 words, medium < 30 word, large < 50, very_large
        labeled_size_update_params = [[{MONGO_DB_CONSTANTS.N_WORDS_FIELD : {"$lte" : 15}}, {'$set': {MONGO_DB_CONSTANTS.LABEL_SIZE_FIELD: SIZE_CONSTANTS.SMALL}}],
                  [{MONGO_DB_CONSTANTS.N_WORDS_FIELD  : {"$lte" : 30, '$gt' : 15}}, {'$set': {MONGO_DB_CONSTANTS.LABEL_SIZE_FIELD : SIZE_CONSTANTS.MEDIUM}}],
                  [{MONGO_DB_CONSTANTS.N_WORDS_FIELD  : {"$lte" : 50, '$gt' : 30}}, {'$set': {MONGO_DB_CONSTANTS.LABEL_SIZE_FIELD : SIZE_CONSTANTS.LARGE}}],
                  [{MONGO_DB_CONSTANTS.N_WORDS_FIELD  : {'$gt' : 50}}, {'$set': {MONGO_DB_CONSTANTS.LABEL_SIZE_FIELD : SIZE_CONSTANTS.VERY_LARGE}}],
                ]

        for update_query, update_operation in labeled_size_update_params :
            mongo_clean_client.update_collection(update_query, update_operation)
         
        # 5 Remove duplicates
        pipeline_remove_duplicate = [{'$group' : {MONGO_DB_CONSTANTS.ID_FIELD: '$Text', 'first_document_id' : {'$first' : '$_id'}}}]
        mongo_clean_client.execute_aggregate_pipeline_operation(pipeline_remove_duplicate, remove_duplicates_texts)
    
    except Exception as e : 
      print('An error occured : ')
      print(e)
      sys.exit(1)