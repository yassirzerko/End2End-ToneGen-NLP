import os
from src.core.constants import FEATURE_FORMAT_CONSTANTS
import csv

def get_sub_folders_names(root_folder_path) :
    return [folder.name for folder in os.scandir(root_folder_path) if folder.is_dir()]

def calculate_global_accuracy(file_path) :
    file = open(file_path, 'r')
    reader = csv.reader(file)
    next(reader)

    total_row = 0
    correct_preds = 0
    for row in reader : 
        if len(row) == 0 : 
            continue
     
        pred, label = row[1:3]
        if pred == label :
            correct_preds += 1
        total_row += 1
    
    file.close()
    return correct_preds / total_row


if __name__ == '__main__' :
    '''
    Collect data about training and testing fold validation and get the accuracy of the best model for each combination of model, feature vector format, and its path.                                                                                                                                  
    '''

    # Cree un fichier csv
    # chaque ligne c'est un model, son nom, sa methode, son accuracy general et son path
    # puis faire un deuxieme fichier pour chaque model trouver le meilleur pour chauqe configuration --> c'est ce qu'on le serveur retourne au clienr
    output_folder_path  = 'output_folder'
    output_file_path = 'training_summary.csv'
    columns_names = ['model_name', 'feature_format', 'accuracy', 'path', 'split']
    output_file = open(output_file_path, 'w+', newline='')
    writer = csv.writer(output_file)
    writer.writerow(columns_names)


    for feature_format in [FEATURE_FORMAT_CONSTANTS.BOW, FEATURE_FORMAT_CONSTANTS.TF_IDF, FEATURE_FORMAT_CONSTANTS.W2V_MAX, FEATURE_FORMAT_CONSTANTS.W2V_MEAN,FEATURE_FORMAT_CONSTANTS.W2V_SUM] :
        feature_format_folder_path = os.path.join(output_folder_path, feature_format)
        model_names = get_sub_folders_names(feature_format_folder_path)


        for model_name in model_names:
            model_folder_path = os.path.join(feature_format_folder_path, model_name)
            split_folders_names = get_sub_folders_names(model_folder_path)

            for split_idx, split_folder_name in enumerate(split_folders_names) :
                split_model_folder_path = os.path.join(model_folder_path, split_folder_name)
                predictions_file_path = os.path.join(split_model_folder_path, 'predictions.csv')
                model_path = [os.path.join(split_model_folder_path,file) for file in os.listdir(split_model_folder_path) if file.endswith('.pkl')][0]
                accuracy = calculate_global_accuracy(predictions_file_path)
                
                writer.writerow([model_name, feature_format, accuracy, model_path, split_idx])


    
    output_file.close()









