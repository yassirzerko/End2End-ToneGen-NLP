const featureFormatNameConversion = {
    "Bag of Visual Words (BoVW)":'bow',
    "Term Frequency-Inverse Document Frequency (TF-IDF)" :"tf-idf",
    "Word2Vec - Sum of Each Dimension":"w2v-sum",
    "Word2Vec - Mean of Each Dimension" :"w2v-mean",
    "Word2Vec - Max of Each Dimension":"w2v-max"

}
export const get_accuracy_sentence = (modelsData, modelName, featureFormat) => {
    for (let idx in modelsData) {
      
      if (modelsData[idx]['model_name'] == modelName) {
        return `With the choosen configuration, with the model ${modelName} and the feature representation ${featureFormat}, the model had an accuracy of ${modelsData[idx][featureFormatNameConversion[featureFormat]]} on the test set.`
      }
    } 
  }