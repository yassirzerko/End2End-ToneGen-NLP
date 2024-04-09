from openai import OpenAI
from src.core.constants import OpenAIConstants, TONES_CONSTANTS

# API_doc: https://platform.openai.com/docs/introduction
class GPT_Text_Generator :
  """
    A class for generating text using the OpenAI GPT model.

    Attributes:
    - api_key (str): The API key used for accessing the OpenAI GPT model.

    Methods:
    - __init__(api_key): Constructor method to initialize the GPT_Text_Generator instance with the provided API key.
    - connect_to_client(): Connects to the OpenAI client using the provided API key.
    - __make_request(tone, size, n_texts, gpt_model_name): Makes a request to the OpenAI GPT model to generate text based on the provided parameters.
    - generate_paragraphs(tone, size, n_required_paragraph, gpt_model_name): Generates paragraphs of text for the specified tone, size, and number of required paragraphs using the OpenAI GPT model.
    - generate_prompt(): Generates a prompt to be used in the conversation with the GPT model.
  """

  @staticmethod
  def generate_prompt() :
    tones = ', '.join(TONES_CONSTANTS.TONES)
    context = f'Given the following sentence tones: {tones}.\n'
    role = 'You are used as a data generator to create a database with texts from each of the 10 tones. The data will be used afterward to train a tone detection model using NLP.\n'

    rules = 'Here are the rules:\n'
    rules += '1. Each time the user interacts with you, they will form a message with 3 values (A, B, C). The message will always be in the form: A, B, C.\n'
    rules += '2. A refers to the tone, B to the number of texts to generate, and C to the size of the texts to generate.\n'
    rules += '3. After each interaction, your response should consist of exactly B texts that are in tone A and are of size C. Each text should be separated by the character "|".\n'
    rules += '4. C can have 4 values: "small" (one short sentence), "medium" (2 to 3 sentences), "large" (4 to 6 sentences), "very-large" (more than 6 sentences).\n'
    rules += '5. Your response should only contain the texts separated by "|", nothing else (no added explanation, no introduction of the response, just the texts).\n'
    rules += '6. If you cannot generate the required number of texts B of size C within a single response, fit as many texts that respect the conditions A and C as possible.\n'
    rules += '7. Try varying the subject and the tense of the texts.\n'

    return context + role + rules

  def __init__(self, api_key) :
    self.api_key = api_key

  def connect_to_client(self) :
    self.client = OpenAI(api_key = self.api_key)

  # Max response size limit approx 4000 words
  def __make_request(self, tone, size, n_texts, gpt_model_name) :
    """
        Makes a request to the OpenAI GPT model to generate text based on the provided parameters.

        Args:
        - tone (str): The tone of the text.
        - size (str): The size of the text (e.g., 'small', 'medium', 'large').
        - n_texts (int): The number of texts to generate.
        - gpt_model_name (str): The name of the GPT model to use for generation.

        Returns:
        - Response: The response object containing the generated text.
    """

    prompt = GPT_Text_Generator.generate_prompt()
    parameters = f'{tone}, {n_texts}, {size}'
    response = self.client.chat.completions.create(
        model = gpt_model_name,
        messages = [
            {'role' : OpenAIConstants.SYSTEM_ROLE, 'content' : prompt},
            {'role' : OpenAIConstants.USER_ROLE, 'content' : parameters}
        ])

    return response

  def generate_paragraphs(self, tone, size, n_required_paragraph, gpt_model_name) :
    """
        Generates paragraphs of text for the specified tone, size, and number of required paragraphs using the OpenAI GPT model.

        Args:
        - tone (str): The tone of the text.
        - size (str): The size of the text (e.g., 'small', 'medium', 'large').
        - n_required_paragraph (int): The number of paragraphs to generate.
        - gpt_model_name (str): The name of the GPT model to use for generation.

        Yields:
        - list: A list of generated texts for each iteration.
    """
    
    n_paragraph_generated = 0
    request_number = 0

    while n_paragraph_generated < n_required_paragraph :
      remaining_paragraph_to_generate = n_required_paragraph - n_paragraph_generated
      print(f'Request number {request_number},  remaining number of texts to generate : {remaining_paragraph_to_generate}.')
      remaining_paragraph_to_generate = min(remaining_paragraph_to_generate, 50)
      response = self.__make_request(tone, size, remaining_paragraph_to_generate, gpt_model_name)

      texts = [text for text in response.choices[0].message.content.strip().split('|') if text != '']

      n_paragraph_generated += len(texts)
      print(f'{n_paragraph_generated} generated texts after request {request_number}')
      request_number += 1

      yield texts
    


