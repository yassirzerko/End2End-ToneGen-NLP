# End2End Machine learning project: From automated text generation to deployed tone classifier

## Overview
This project aims to generate text of various sizes and tones using the OpenAI API, store the generated texts in MongoDB Atlas, analyze and clean them, apply different word embedding NLP techniques (Bag of Words, TF-IDF, Word2Vec) with various models, and finally deploy the best-performing model as a Flask API and provide access to it throught a simple `React` client. The **end goal** is to predict the tone of input text.

## Features
- Generate over 8500 text of different tones and sizes using OpenAI API.
- Store the generated texts in MongoDB Atlas and handle duplication and generation errors.
- Generate balanced datasets for **20 text embedding techniques** including Bag-of-Words (BoW), TF-IDF, 3 Word2Vec text embedding methode (max, mean and sum pooling) 
    with 5 different Word2Vec models for words embedding, and 3 BERT models for text embedding.
- Perform hyperparameter search for various models for all text embedding datasets and store the predictions and models.
- Deploy the best-performing model as a Flask API for tone prediction.
- Provide a user-friendly interface for the tone predictor through a simple React application.

## Scripts 
The project comprises various distinct scripts, each serving a specific purpose. Here are their functionalities.

| Script Name                                   | Script Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| src.script.text_generation.generate_texts    | Generate data by creating paragraphs of text with `OpenAi` API for each tone and inserting them into a MongoDB database.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| src.script.text_generation.clean_generated_texts | Clean and preprocess text data stored in a MongoDB collection. This scripts performs data cleaning and preprocessing operations. The operations include transferring data from the raw collection to the clean collection, splitting texts containing numbered patterns into separate entries, updating the word count for each text entry, adding labeled sizes based on word count thresholds, and removing duplicate text entries except for the first occurrence.                                                                                                                                                                                                                                                                                                                                                  |
| src.scripts.ml.generate_text_embedding_datasets  | Generate balanced split datasets with multiple text emebedding (text feature representation) techniques for machine learning tasks : Bag-of-Words (BoW), TF-IDF, 3 Word2Vec text embedding methode (max, mean and sum pooling)  with 5 different Word2Vec models for words embedding, and 3 BERT models for text embedding for a total of **20 text embedding techniques** . It generates 20 CSV training datasets and 20 CSV testing datasets for each split. The resulting datasets can be used to train models. In addition, download Word2Vec and Bert converters if they don't exist.                                                                                                                                                                                                                                                                                                                                                                                        |
| src.scripts.ml.train_optimize_save_models        | Train, test, and save machine learning models concurrently (one process for each model) with different parameters and text embedding techniques with k-fold cross-validation using the CSV-generated dataset. It also generate an additionnal CSV file where the summary of the trained models is saved, this file is necesary for the flask Server.                                                                                                                                                                                              

These scripts form a pipeline for generating, cleaning, and processing text data, training machine learning models, and saving the trained models. 
1. **generate_texts**: Generates raw text data and stores it in a MongoDB database.
2. **clean_generated_texts**: Cleans and preprocesses the raw text data stored in the MongoDB collection.
3. **generate_text_embedding_datasets**: Generates balanced split datasets with various text embedding techniques for machine learning tasks.
4. **train_optimize_save_models**: Trains machine learning models concurrently with different parameters and text embedding techniques, optimizes them, and saves the trained models along with a summary CSV file necessary for the Flask server.

At the end of the pipeline, the data required by the Flask server is ready, and the server can be deployed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
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

## Text Embedding Techniques

**General Explanation**

Text embedding techniques transform text into dense vector representations that capture semantic meanings. In this project, a variety of text embedding techniques are employed to characterize texts. Here are the text embedding techniques used in the project: Bag-of-Words (BoW), TF-IDF, and 3 Word2Vec text representations (max, mean, sum) with 5 different Word2Vec models, as well as 2 BERT models and 1 RoBERTa model, for a total of **20 text embeddings**.

**Bag of Words (BoW)**

Represent text by counting occurrences of word in a predefined vocabulary.

**Term Frequency-Inverse Document Frequency (TF-IDF)**

Weigh each word by its frequency in the document and its rarity across documents.

**Each Dimension sum Pooling**

Calculate the sum of word embeddings across each dimension in the Word2Vec space

**Each Dimension mean Pooling**

Compute the average of word embeddings across each dimension in the Word2Vec space, followed by pooling.

**Each Dimension max Pooling**

Determine the maximum value of word embeddings across each dimension in the Word2Vec space, followed by pooling.

**BERT Models**

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that generates context-aware embeddings for each token in the input text. Two different BERT models are used in this project for text embedding.

**RoBERTa Model**

RoBERTa (Robustly optimized BERT approach) is another variant of the BERT model that employs longer training with larger batches and more data. It provides enhanced performance compared to the original BERT model.

Each of these text embedding techniques captures different aspects of the input text, allowing for a comprehensive representation suitable for various machine learning tasks.

## Setup
1. **Ensure Python and pip are installed on your system.**
2. **Install the required dependencies in the root folder:**
   ```bash
   pip install -r requirements.txt
3. **Copy the `.env-template` file and rename the copy to `.env` and fill the required values in the new `.env` file.**
   - This step is required if you want to use the generation data features (generating texts and saving in MongoDB Atlas). 
   - Make sure you have an OpenAI API key and a MongoDB URI to fill in the `.env` file.

