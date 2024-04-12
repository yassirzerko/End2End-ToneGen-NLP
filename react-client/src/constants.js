export const appExplanation = `
Welcome to our Tone Prediction Application! 
Choose a trained model and text embedding technique from the dropdown menus, then input your text to predict its tone.
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
  ["General explanation", "Text embedding techniques transform text into dense vector representations that capture semantic meanings. A variety of 20 text embedding techniques offered."],
  ["Bag of Words (BoW)", "Represent text by counting occurrences of words in a predefined vocabulary."],
  ["Term Frequency-Inverse Document Frequency (TF-IDF)", "Weigh each word by its frequency in the document and its rarity across documents."],
  ["Word2Vec - Each Dimension Pooling", "Calculate the max/average/sum word embeddings across each dimension in the Word2Vec space."],
  ["BERT Models"," Transformer-based model that generates context-aware embeddings for texts."],
  ["RoBERTa","Variant of the BERT model that employs more data. It provides enhanced performance compared to the original BERT model."]
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

export const server_URI = 'http://127.0.0.1:5000/'