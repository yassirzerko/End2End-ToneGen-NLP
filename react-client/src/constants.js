export const appExplanation = `
Welcome to our Tone Prediction Application! 
Choose a trained model and feature representation technique from the dropdown menus, then input your text to predict its tone.
Tones Available for Prediction:
Formal,
Casual,
Serious,
Humorous,
Professional,
Instructive,
Persuasive,
Narrative,
Analytical,
Empathetic.
Get started now and discover the tone of your text!
`;

export const featureRepresentations = [
  ["General explanation", "Feature representation techniques transform text into a format that machine learning models can understand. In this project, five distinct methods of feature representation are employed to characterize texts."],
  ["Bag of Visual Words (BoVW)", "Represent text by counting occurrences of visual word 'clusters' in a predefined vocabulary."],
  ["Term Frequency-Inverse Document Frequency (TF-IDF)", "Weigh each word by its frequency in the document and its rarity across documents."],
  ["Word2Vec - Sum of Each Dimension", "Calculate the sum of word embeddings across each dimension in the Word2Vec space."],
  ["Word2Vec - Mean of Each Dimension", "Compute the average of word embeddings across each dimension in the Word2Vec space."],
  ["Word2Vec - Max of Each Dimension", "Determine the maximum value of word embeddings across each dimension in the Word2Vec space."]
];

export const modelDescriptions = [
  ["General explanation", "Various machine learning models are employed to predict the tone of input texts. In this project, a selection of models is utilized, each offering unique approaches to tone classification."],
  ["Random Forest", "A versatile ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification problems."],
  ["Gradient Boosting", "An ensemble learning technique that builds a strong model by iteratively adding weak learners (decision trees) to minimize the loss function."],
  ["Histogram Gradient Boosting", "Similar to Gradient Boosting, this method builds an ensemble of decision trees, but uses histogram-based techniques for faster training and better accuracy."],
  ["AdaBoosting", "A boosting algorithm that combines multiple weak classifiers to form a strong classifier by focusing on the mistakes of the previous classifiers."],
  ["Neural Network", "A deep learning model inspired by the structure of the human brain, consisting of interconnected nodes organized in layers. It's capable of learning complex patterns in data."],
];

const models_data = [{id : 'id1', model_name: 'rf', bow_acc : 0.90,tf_idf_acc : 0.94, w2v_max_acc : 0.93, w2v_sum_acc : 0.40,  w2v_sum_mean : 0.60,},
{id : 'id2', model_name: 'gb', bow_acc : 0.90,tf_idf_acc : 0.94, w2v_max_acc : 0.93, w2v_sum_acc : 0.40,  w2v_sum_mean : 0.60,},
{id : 'id3', model_name: 'mlp', bow_acc : 0.90,tf_idf_acc : 0.94, w2v_max_acc : 0.93, w2v_sum_acc : 0.40,  w2v_sum_mean : 0.60,},
{id : 'id4', model_name: 'ada', bow_acc : 0.90,tf_idf_acc : 0.94, w2v_max_acc : 0.93, w2v_sum_acc : 0.40,  w2v_sum_mean : 0.60,},]

export const Mock_data = {models : models_data , feature_format : ['bow', 'tf_idf', 'w2v_max', 'w2v_sum', 'w2v_mean']} 

export const API = 'api_test'