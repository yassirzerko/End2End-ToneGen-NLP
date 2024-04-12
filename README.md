# End2End ML PROJECT: From text generation to deployed tone classifier

## Overview
This project aims to generate text of various sizes and tones using the OpenAI API, store the generated texts in MongoDB Atlas, analyze and clean them, apply different word embedding NLP techniques (Bag of Words, TF-IDF, Word2Vec) with various models, and finally deploy the best-performing model as a Flask API and provide access to it throught a simple `React` client. The **end goal** is to predict the tone of input text.

## Features
- Generate text of different tones and sizes using OpenAI API.
- Store the generated texts in MongoDB Atlas and handle duplication and generation errors.
- Generate balanced datasets for **17 text embedding techniques** including Bag-of-Words (BoW), TF-IDF, and 3 Word2Vec representations (maximum, mean, and sum of each dimension) with 5 different Word2Vec models.
- Perform hyperparameter search with various models for all text embedding datasets and store the predictions and models.
- Deploy the best-performing model as a Flask API for tone prediction.
- Provide a user-friendly interface for the tone predictor through a simple React application.

## Scripts 
The project comprises various distinct scripts, each serving a specific purpose. Here are their functionalities.

| Script Name                                   | Script Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| src.script.text_generation.generate_texts    | Generate data by creating paragraphs of text with `OpenAi` API for each tone and inserting them into a MongoDB database.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| src.script.text_generation.clean_generated_texts.py | Clean and preprocess text data stored in a MongoDB collection. This scripts performs data cleaning and preprocessing operations. The operations include transferring data from the raw collection to the clean collection, splitting texts containing numbered patterns into separate entries, updating the word count for each text entry, adding labeled sizes based on word count thresholds, and removing duplicate text entries except for the first occurrence.                                                                                                                                                                                                                                                                                                                                                  |
| src.scripts.ml.generate_feature_csv_datasets  | Generate balanced split datasets with multiple feature representation techniques for machine learning tasks : Bag-of-Words (BoW), TF-IDF, and 3 Word2Vec text representations (max, mean, sum) with 5 different Word2Vec models, for a total of **17 text representation techniques** . It generates CSV files for each split and each feature representation technique. The resulting fil datasets can be used to train models. In addition, download a word2vec converter if it doesn't exist.                                                                                                                                                                                                                                                                                                                                                                                                                   |
| src.scripts.ml.train_test_save_models        | Train, test, and save machine learning models with different parameters and feature feature representation techniques with k-fold cross-validation using the CSV-generated dataset. It also generate an additionnal CSV file where the summary of the trained models is saved, this file is necesary for the flask Server                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                                                                                        
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

## Feature Representation Techniques

**General Explanation**

Feature representation techniques transform text into a format that machine learning models can understand. In this project, five distinct methods of feature representation are employed to characterize texts. Here are the used feature representation techniques in the project : Bag-of-Words (BoW), TF-IDF, and 3 Word2Vec text representations (max, mean, sum) with 5 different Word2Vec models, for a total of **17 text representation techniques**ag-of-Words (BoW), TF-IDF, and 3 Word2Vec text representations (max, mean, sum) with 5 different Word2Vec models, for a total of **17 text representation techniques** .

**Bag of Visual Words (BoVW)**

Represent text by counting occurrences of visual word 'clusters' in a predefined vocabulary.

**Term Frequency-Inverse Document Frequency (TF-IDF)**

Weigh each word by its frequency in the document and its rarity across documents.

**Sum of Each Dimension**

Calculate the sum of word embeddings across each dimension in the Word2Vec space.

**Mean of Each Dimension**

Compute the average of word embeddings across each dimension in the Word2Vec space.

**Max of Each Dimension**

Determine the maximum value of word embeddings across each dimension in the Word2Vec space.


## Setup
1. **Ensure Python and pip are installed on your system.**
2. **Install the required dependencies in the root folder:**
   ```bash
   pip install -r requirements.txt
3. **Copy the `.env-template` file and rename the copy to `.env` and fill the required values in the new `.env` file.**
   - This step is required if you want to use the generation data features (generating texts and saving in MongoDB Atlas). 
   - Make sure you have an OpenAI API key and a MongoDB URI to fill in the `.env` file.

