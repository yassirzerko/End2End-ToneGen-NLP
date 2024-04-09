import random
import math
from src.core.constants import TONES_CONSTANTS, SIZE_CONSTANTS, MONGO_DB_CONSTANTS
import re

class MlUtils : 
    @staticmethod
    def get_k_folds_balanced_splits_ids(db_client, k = 5) :
        """
        Generates balanced k-fold splits of data IDs for cross-validation.

        Parameters:
        - db_client: Database client object.
        - k: Number of folds for cross-validation (default is 5).

        Returns:
        - List of k splits, where each split is a list containing two lists: train_ids and test_ids.
        """
        folds_ids = MlUtils.get_balanced_folds_ids(db_client, k)
        splits = [ [[], []] for _ in range(k)]

        for n_fold in range(k) :
            splits[n_fold][1] += folds_ids[n_fold]

            for n_fold_s in range(k) :
                if n_fold_s == n_fold : 
                    continue
                
                splits[n_fold][0] += folds_ids[n_fold_s]
        
        return splits

    @staticmethod
    def get_balanced_folds_ids(db_client, k = 5) :
        """
        Retrieves balanced folds of data IDs from a database client.

        Parameters:
        - db_client: Database client object.
        - k: Number of folds for cross-validation (default is 5).

        Returns:
        - List of k lists, where each list contains data IDs for a fold.

        Note:
        The folds are balanced by label (Tone) and by text size.
        """

        folds_ids = [ [] for _ in range (k) ]
        for tone in TONES_CONSTANTS.TONES :
            for size in [SIZE_CONSTANTS.SMALL, SIZE_CONSTANTS.MEDIUM, SIZE_CONSTANTS.LARGE, SIZE_CONSTANTS.VERY_LARGE] :
                data = list(db_client.get_collection_data({MONGO_DB_CONSTANTS.TONE_FIELD : tone, MONGO_DB_CONSTANTS.LABEL_SIZE_FIELD : size}, {MONGO_DB_CONSTANTS.ID_FIELD : 1}))
                random.shuffle(data)
                nb_entries_by_fold = int(len(data) / k)

                for n_fold in range (k) : 
                    nb_ids_to_add = nb_entries_by_fold if  k*nb_entries_by_fold + nb_entries_by_fold <= len(data) else len(data)
                    fold_data = data[k*nb_entries_by_fold : k*nb_entries_by_fold + nb_ids_to_add]

                    folds_ids[n_fold] += [entry[MONGO_DB_CONSTANTS.ID_FIELD] for entry in fold_data]

        return folds_ids
    

    @staticmethod
    def get_word_idf_data(data) :
        """
        Computes the inverse document frequency (IDF) of words in a given dataset.

        Parameters:
        - data: List of data entries.

        Returns:
        - Dictionary containing IDF values for each unique word in the dataset.
        """
        
        words_idf = {}
        n_texts = 0

        for entry in data :
            words = re.split(r'\W+', entry['Text'])
            uniques = list(set(words))
            n_texts  += 1
            for word in uniques :
                if word not in words_idf :
                    words_idf[word] = 0
                words_idf[word] += 1

        words_idf = {word : math.log(n_texts) / (1 + words_idf[word]) for word in words_idf}
        return words_idf
    

