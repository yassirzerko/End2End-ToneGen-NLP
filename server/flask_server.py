import flask
from flask import request
from flask_cors import CORS
from src.core.constants import FEATURE_FORMAT_CONSTANTS, PATH_NAME_CONSTANTS
from src.core.ml.ml_models_utils import MlModelsUtils
from src.core.ml.ml_models_utils import NlpFeaturesUtils
import os 
import csv

# This Python file implements a Flask web application that serves as an API for machine learning models.
# It provides two main routes: '/' for retrieving data about the best-performing models, and '/predict' for making predictions based on input text.
# The file includes functions to load necessary data, retrieve model information, and make predictions, along with initialization of Flask and Flask-CORS.
# Require the outputs of the script src.scripts.ml.train_optimize_save_models

app = flask.Flask(__name__)
CORS(app)

w2v_converters = NlpFeaturesUtils.load_word2vec_converters()
bert_converters = NlpFeaturesUtils.load_bert_converters()

def get_best_models_data(training_summary_file_path) :
    """
    Returns:
    - A dictionary containing the best-performing model data for each combination of model and feature vector format.
    """
    file = open(training_summary_file_path,'r')
    reader = csv.reader(file)
    next(reader) 
    best_models_data = {}
    for row in reader : 
        model_name, uuid, feature_format, accuracy, path, split_idx = row

        if model_name not in best_models_data :
            best_models_data[model_name] = {}
        
        if feature_format not in best_models_data[model_name] :
            best_models_data[model_name][feature_format] = {'accuracy' : 0}

        
        if float(accuracy) >  float(best_models_data[model_name][feature_format]['accuracy']) :
            best_models_data[model_name][feature_format] = { 'accuracy' : accuracy, 'path' : path, 'split_idx'  : split_idx}
    
    file.close()
    return best_models_data
    
@app.route('/')
def get_models_data() : 
    """
    Returns:
    - A list of dictionaries representing the formatted response containing data about the best-performing models.
        Each dictionary includes the model name, feature vector format, and accuracy.

    Description:
    This route function returns data about the best-performing models for each combination of model and feature vector format.
    """
    
    models_data = get_best_models_data(PATH_NAME_CONSTANTS.TRAINED_MODELS_DATA_FILE)
    formatted_response = {}
    training_data = []
    for model_name in models_data :
        model_dict_data = {'model_name' : model_name}
        for feature_format in FEATURE_FORMAT_CONSTANTS.FEATURES_NAMES :
            model_dict_data[feature_format] = round(float(models_data[model_name][feature_format]['accuracy']),2)
        
        training_data.append(model_dict_data)
    
    formatted_response = {'models_data' : training_data, 'feature_names' : FEATURE_FORMAT_CONSTANTS.FEATURES_NAMES}
        
    return formatted_response

@app.route('/predict', methods =['GET'])
def make_prediction() :
    print('in')
    """
    Returns:
    - The prediction result using the best-performing model for the specified model name and feature vector format.
    """
    
    model_name = request.args['model_name']
    feature_vec = request.args['feature_format']
    input_text = request.args['input_text']
    models_data = get_best_models_data(PATH_NAME_CONSTANTS.TRAINED_MODELS_DATA_FILE)
    idf_file_path, voc_file_path = NlpFeaturesUtils.get_idf_voc_path(models_data[model_name][feature_vec]['split_idx'])
    prediction = MlModelsUtils.use_trained_model_to_predict_tone(models_data[model_name][feature_vec]['path'], input_text, feature_vec, idf_file_path, voc_file_path, w2v_converters, bert_converters)
    return {'prediction' : prediction}
    
