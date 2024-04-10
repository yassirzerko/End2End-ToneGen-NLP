import random
from src.core.constants import TONES_CONSTANTS, OpenAIConstants, SIZE_CONSTANTS, ENV_CONSTANTS, MONGO_DB_CONSTANTS
from src.core.text_generation.gpt_text_generator import GPT_Text_Generator
from src.core.text_generation.mongo_client import Mongo_Client
from src.core.read_env import get_env_variables
import sys
import traceback


def generate_texts(text_generator, mongo_client, n_required_paragraph_by_tone, size, gpt_model_name) :
    """
    Generate data by generating paragraphs of text for each tone and inserting them into a MongoDB database.

    Args:
    - text_generator (TextGenerator): An instance of the TextGenerator class responsible for generating text.
    - mongo_client (MongoClient): An instance of the MongoClient class for interacting with MongoDB.
    - n_required_paragraph_by_tone (int): The number of paragraphs to generate for each tone.
    - size (str): The size of the generated text (e.g., 'small', 'medium', 'large').
    - gpt_model_name (str): The name of the GPT model used for generating text.

    Returns:
    None
    """

    tones = random.sample(TONES_CONSTANTS.TONES, len(TONES_CONSTANTS.TONES))
    for tone in tones :
      generated_text = 0
      print(f'STARTING : TONE : {tone}, n_texts : {n_required_paragraph_by_tone}, size : {size}')
      for texts in text_generator.generate_paragraphs(tone, size, n_required_paragraph_by_tone, gpt_model_name) :

        data = [{MONGO_DB_CONSTANTS.TEXT_FIELD : text.strip(), MONGO_DB_CONSTANTS.TONE_FIELD: tone, MONGO_DB_CONSTANTS.N_WORDS_FIELD : len(text.strip().split(' ')), MONGO_DB_CONSTANTS.SIZE_FIELD: size, MONGO_DB_CONSTANTS.GENERATED_BY_FIELD: gpt_model_name} for text in texts]
        mongo_client.insert_data(data)
        generated_text += len(data)

      print(f'END : TONE : {tone}, n_texts : {n_required_paragraph_by_tone}, size : {size}, generated {generated_text} texts. ')
      print(10 * '---')
      print('\n')
    

if __name__ == '__main__' :
    """
    Generate data by generating paragraphs of text for each tone and inserting them into a MongoDB database.
    Retrieves environment variables, initializes GPT text generator and MongoDB client, and generates data for specified combinations of texts and sizes.
    """

    # Decomment the one you want to use
    gpt_model_name = OpenAIConstants.GPT_3_MODEL_NAME
    # gpt_model_name = OpenAIConstants.GPT_4_MODEL_NAME)
    
    # Fill your inputs_combinations
    # Example with 100 texts from each of the 4 defined sizes
    inputs_combinations = [(100, SIZE_CONSTANTS.SMALL), (100, SIZE_CONSTANTS.MEDIUM), (100, SIZE_CONSTANTS.LARGE), (100, SIZE_CONSTANTS.VERY_LARGE)]

    # Retrieve environment variables
    env_variables, error, error_msg = get_env_variables() 
    if error : 
      print('An error occured : ')
      print(error_msg)
      sys.exit(1)
    
    
    try : 
      # Initialize GPT text generator
      text_generator = GPT_Text_Generator(env_variables[ENV_CONSTANTS.OPEN_AI_API_FIELD])
      text_generator.connect_to_client()

      # Initialize MongoDB client
      mongo_client = Mongo_Client(env_variables[ENV_CONSTANTS.MONGO_URI_FIELD])
      mongo_client.connect_to_db(env_variables[ENV_CONSTANTS.MONGO_DB_NAME_FIELD], env_variables[ENV_CONSTANTS.DB_RAW_COLLECTION_FIELD])

      # Generate data for each combination of texts and sizes
      for (n_texts, size) in inputs_combinations :
        generate_texts(text_generator, mongo_client, n_texts, size, gpt_model_name)
    
    except Exception as e : 
      print('An error occured : ')
      print(e)
      traceback_info = traceback.format_exc()
      print(traceback_info)
      sys.exit(1)
       
