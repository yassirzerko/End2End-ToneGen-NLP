import flask
from flask import request
from flask_cors import CORS
from src.core.constants import FEATURE_FORMAT_CONSTANTS
from src.core.ml.nlp_features_utils import NlpFeaturesUtils
from src.core.ml.ml_models_utils import MlModelsUtils
import os 
import csv

app = flask.Flask(__name__)
CORS(app)
w2v_converter_path = 'glove-wiki-300-w2v.wordvectors'

def get_idf_file_path(split_idx) :
    idf_file_path = os.path.join('datasets', f'split{split_idx}', 'tf-idf-data.json')
    return idf_file_path

def get_best_models_data() :
    """
    Returns:
    - A dictionary containing the best-performing model data for each combination of model and feature vector format.
    """
    
    training_summary_file_path = 'trained_models_data.csv'
    file = open(training_summary_file_path,'r')
    reader = csv.reader(file)
    next(reader) 
    best_models_data = {}
    for row in reader : 
        model_name, feature_format, accuracy, path, split_idx = row

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
    
    models_data = get_best_models_data()
    formatted_response = []
    for model_name in models_data :
        model_dict_data = {'model_name' : model_name}
        for feature_format in [FEATURE_FORMAT_CONSTANTS.BOW, FEATURE_FORMAT_CONSTANTS.TF_IDF, FEATURE_FORMAT_CONSTANTS.W2V_MAX, FEATURE_FORMAT_CONSTANTS.W2V_MEAN, FEATURE_FORMAT_CONSTANTS.W2V_SUM] :
            model_dict_data[feature_format] = round(float(models_data[model_name][feature_format]['accuracy']),2)
        
        formatted_response.append(model_dict_data)
        
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
    models_data = get_best_models_data()
    idf_file_path = get_idf_file_path(models_data[model_name][feature_vec]['split_idx'])
    prediction = MlModelsUtils.use_trained_model_to_preidct_tone(models_data[model_name][feature_vec]['path'], input_text, feature_vector_format = feature_vec, w2v_model_path = w2v_converter_path, idf_data_path = idf_file_path)
    return {'prediction' : prediction}
    
