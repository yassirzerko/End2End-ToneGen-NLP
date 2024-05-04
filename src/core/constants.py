class OpenAIConstants : 
    SYSTEM_ROLE = 'system'
    USER_ROLE = 'user'

    GPT_3_MODEL_NAME = 'gpt-3.5-turbo'
    GPT_4_MODEL_NAME = 'gpt-4'

class TONES_CONSTANTS : 
    TONES_WITH_DESCRIPTION = [
        ["Formal", "Uses professional language and adheres to standard grammar and syntax conventions. Suitable for academic papers, business communications, and official documents."],
        ["Casual", "Uses informal language and colloquial expressions. Suitable for friendly conversations, social media posts, and informal writing."],
        ["Serious", "Conveys a somber or solemn tone, often used for discussing sensitive or grave topics. Suitable for news reports, memorials, and serious discussions."],
        ["Humorous", "Conveys humor and wit, often through puns, wordplay, and lighthearted commentary. Suitable for comedy sketches, satire, and entertainment content."],
        ["Professional", "Similar to formal tone but with a focus on professionalism and expertise. Suitable for resumes, professional profiles, and business presentations."],
        ["Instructive", "Provides clear and concise instructions or guidance on a specific topic or task. Suitable for tutorials, guides, and instructional materials."],
        ["Persuasive", "Aims to convince or persuade the audience to adopt a particular viewpoint or take action. Often uses rhetorical devices and persuasive techniques. Suitable for marketing campaigns, persuasive essays, and speeches."],
        ["Narrative", "Tells a story or recounts events in a sequential manner. Often includes descriptive language and vivid imagery to engage the reader. Suitable for storytelling, fiction writing, and narrative essays."],
        ["Analytical", "Analyzes information or data in a logical and systematic manner, often presenting findings or insights. Suitable for research papers, analytical essays, and data analysis reports."],
        ["Empathetic", "Conveys understanding, compassion, and empathy towards the audience or subject matter. Suitable for comforting messages, support forums, and counseling sessions."]
    ]

    TONES = [tone[0] for tone in TONES_WITH_DESCRIPTION]

class MONGO_DB_CONSTANTS : 
    ID_FIELD = "_id"
    RAW_ID_FIELD = "raw_collection_id"
    TONE_FIELD = "Tone"
    TEXT_FIELD = "Text"
    N_WORDS_FIELD = "N_words"
    SIZE_FIELD = "Generated_size"
    GENERATED_BY_FIELD = "Generated_by"
    LABEL_SIZE_FIELD = "labeled_size"


class SIZE_CONSTANTS : 
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very-large"

class ENV_CONSTANTS : 
    ENV_FILE_NAME = ".env"
    OPEN_AI_API_FIELD = "OPENAI_API_KEY"
    MONGO_URI_FIELD = "MONGO_URI"
    MONGO_DB_NAME_FIELD = "MONGO_DB_NAME"
    DB_RAW_COLLECTION_FIELD = "MONGO_RAW_COLLECTION_NAME"
    DB_CLEAN_COLLECTION_FIELD = "MONGO_CLEAN_COLLECTION_NAME"

W2V_MODEL_NAMES = ['glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', 'glove-wiki-gigaword-300']
BERT_MODELS_NAMES = ['bert-base-uncased','bert-large-uncased', 'roberta-base']

class FEATURE_FORMAT_CONSTANTS:
    BOW = "bow"
    TF_IDF = "tf-idf"
    W2V_MAX = "w2v-max"
    W2V_SUM = "w2v-sum"
    W2V_MEAN = "w2v-mean"
    BERT = "bert"
    W2V_FEATURES = []

    for w2v_model_name in W2V_MODEL_NAMES :
        for op_name in ['max', 'sum', 'mean'] :
            W2V_FEATURES += [ f'w2v_{w2v_model_name}_{op_name}'  ]

    FEATURES_NAMES = [BOW, TF_IDF]  + W2V_FEATURES + BERT_MODELS_NAMES

class PATH_NAME_CONSTANTS :
    GENERATED_DATASETS = 'datasets'
    TRAINED_MODELS = 'trained_models'
    TRAINED_MODELS_DATA_FILE = 'trained_models_data.csv'

