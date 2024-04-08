# End2End ML PROJECT: From text generation to deployed tone classifier

## Overview
This project aims to generate text of various sizes using the OpenAI API, store the generated texts in MongoDB Atlas, analyze and clean them, apply different word embedding NLP techniques (Bag of Words, TF-IDF, Word2Vec, Positional Encoding) with various models, and finally deploy the best-performing model as a Flask API. The end goal is to predict the tone of input text.

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
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up MongoDB Atlas and configure the database connection.
4. Update the OpenAI API key in the code.
5. Run the project.

## Usage
1. Run the script to prompt and generate texts using the OpenAI API.
2. The generated texts will be stored in MongoDB Atlas.
3. Analyze and clean the generated texts.
4. Test different word embedding NLP techniques and models.
5. Deploy the best-performing model as a Flask API.
