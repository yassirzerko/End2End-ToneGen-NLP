# End2End ML PROJECT: From text generation to deployed tone classifier

## Overview
This project aims to generate text of various sizes using the OpenAI API, store the generated texts in MongoDB Atlas, analyze and clean them, apply different word embedding NLP techniques (Bag of Words, TF-IDF, Word2Vec, Positional Encoding) with various models, and finally deploy the best-performing model as a Flask API. The **end goal** is to predict the tone of input text.

## Features
- Generate text of different sizes using OpenAI API.
- Store generated texts in MongoDB Atlas.
- Analyze and clean generated texts, handling duplication and generation errors, in a separate MongoDB database.
- Test multiple word embedding NLP techniques (Bag of Words, TF-IDF, Word2Vec, Positional Encoding) with various models.
- Deploy the best-performing model as a Flask API for tone prediction.

## Tones
The following 10 tones are considered for tone classification:
1. Formal
2. Casual
3. Serious
4. Humorous
5. Professional
6. Instructive
7. Persuasive
8. Narrative
9. Analytical
10. Empathetic

## Setup
1. **Ensure Python and pip are installed on your system.**
2. **Install the required dependencies in the root folder:**
   ```bash
   pip install -r requirements.txt
3. **Copy the `.env-template` file and rename the copy to `.env` and fill the required values in the new `.env` file.**
   - This step is required if you want to use the generation data features (generating texts and saving in MongoDB Atlas). 
   - Make sure you have an OpenAI API key and a MongoDB URI to fill in the `.env` file.

