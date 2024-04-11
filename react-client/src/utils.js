import { server_URI } from "./constants" 

const featureFormatNameConversion = {
    "Bag of Visual Words (BoVW)":'bow',
    "Term Frequency-Inverse Document Frequency (TF-IDF)" :"tf-idf",
    "Word2Vec - Sum of Each Dimension":"w2v-sum",
    "Word2Vec - Mean of Each Dimension" :"w2v-mean",
    "Word2Vec - Max of Each Dimension":"w2v-max"
}

const modelNameConversion = {
  "Random Forest" : "RandomForestClassifier", 
  "Gradient Boosting" : "GradientBoostingClassifier",
  "Histogram Gradient Boosting" : "HistGradientBoostingClassifier", 
  "AdaBoosting" : "AdaBoostClassifier",
  "Neural Network" : "MLPClassifier"
}

export const get_accuracy_sentence = (modelsData, modelName, featureFormat) => {
    for (let idx in modelsData) {
      console.log(modelName)
      if (modelsData[idx]['model_name'] == modelNameConversion[modelName]) {
        return `With the choosen configuration, with the model ${modelName} and the feature representation ${featureFormat}, the model had an accuracy of ${modelsData[idx][featureFormatNameConversion[featureFormat]]} on the test set.`
      }
    } 
  }

export const buildRequest = (model_name, feature_format, input_text) => {
  const url = new URL(server_URI + '/predict')
  model_name = modelNameConversion[model_name]
  url.searchParams.set('model_name', model_name)
  url.searchParams.set('feature_format', featureFormatNameConversion[feature_format])
  url.searchParams.set('input_text', input_text)
  return url.toString()
}