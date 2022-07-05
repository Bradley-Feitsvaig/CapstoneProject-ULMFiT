import os
import shelve

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

"""
ModelShelveDb get,set,and update shelves file that contains dictionary which are used in Model class
"""


class ModelShelveDb:
    def __init__(self):
        super(ModelShelveDb, self).__init__()

    def get_url_for_embedding(self):
        sh = shelve.open(ROOT_DIR + "\\database\\url_for_embedding_ModelShelveDb")
        url_for_embedding = sh['url_for_embedding']
        sh.close()
        return url_for_embedding

    def set_url_for_embedding(self):
        sh = shelve.open(ROOT_DIR + "\\database\\url_for_embedding_ModelShelveDb")
        sh['url_for_embedding'] = {
            "LaBSE_15": ["https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang_preprocess/1",
                         "https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1"],
            "LaBSE_109": ["https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2",
                          "https://tfhub.dev/google/LaBSE/2"],
            "Bert": ["https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
                     "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"],
        }
        sh.close()

    def update_url_for_embedding(self, url_for_embedding):
        sh = shelve.open(ROOT_DIR + "\\database\\url_for_embedding_ModelShelveDb")
        sh['url_for_embedding'] = url_for_embedding
        sh.close()


"""
DatasetPreprocessShelveDb get,set,and update shelves file that contains dictionarys which are used in DatasetPreprocess class
"""


class DatasetPreprocessShelveDb:
    def __init__(self):
        super(DatasetPreprocessShelveDb, self).__init__()

    def get_sentiment_dict(self):
        sh = shelve.open(ROOT_DIR + "\\database\\sentiment_dict_DatasetPreprocessShelveDb")
        sentiment_dict = sh['sentiment_dict']
        sh.close()
        return sentiment_dict

    def set_sentiment_dict(self):
        sh = shelve.open(ROOT_DIR + "\\database\\sentiment_dict_DatasetPreprocessShelveDb")
        sh['sentiment_dict'] = {
            "neutral": 1,
            "negative": 0,
            "positive": 2
        }
        sh.close()

    def update_sentiment_dict(self, sentiment_dict):
        sh = shelve.open(ROOT_DIR + "\\database\\sentiment_dict_DatasetPreprocessShelveDb")
        sh['sentiment_dict'] = sentiment_dict
        sh.close()

    def get_regex_for_language(self):
        sh = shelve.open(ROOT_DIR + "\\database\\regex_for_language_DatasetPreprocessShelveDb")
        regex_for_language = sh['regex_for_language']
        sh.close()
        return regex_for_language

    def set_regex_for_language(self):
        sh = shelve.open(ROOT_DIR + "\\database\\regex_for_language_DatasetPreprocessShelveDb")
        sh['regex_for_language'] = {
            "english": '[^a-zA-Z .]',
            "russian": '[^А-я .]'
        }
        sh.close()

    def update_regex_for_language(self, regex_for_language):
        sh = shelve.open(ROOT_DIR + "\\database\\regex_for_language_DatasetPreprocessShelveDb")
        sh['regex_for_language'] = regex_for_language
        sh.close()


"""
AppShelveDb get,set,and update shelves file that contains dictionarys which are used in App class
"""


class AppShelveDb:
    def __init__(self):
        super(AppShelveDb, self).__init__()

    def get_model_task_dict(self):
        sh = shelve.open(ROOT_DIR + "\\database\\model_task_dict_AppShelveDb")
        model_task_dict = sh['model_task_dict']
        sh.close()
        return model_task_dict

    def set_model_task_dict(self):
        sh = shelve.open(ROOT_DIR + "\\database\\model_task_dict_AppShelveDb")
        sh['model_task_dict'] = {
            'English with LSTM wiki2': {'tasks': ['Sentiment analysis'],
                                        'base language model': "\\savedModels\\englishLSTMNextWord",
                                        'Sentiment analysis model': "\\savedModels\\englishLSTMsentimental",
                                        'Sentiment analysis dataset': "\\datasets\\Twitter US Airline Sentiment "
                                                                      "dataset.csv",
                                        'language': 'english'
                                        },
            'English with GRU wiki2': {'tasks': ['Sentiment analysis'],
                                       'base language model': "\\savedModels\\englishGRUNextWord",
                                       'Sentiment analysis model': "\\savedModels\\englishGRUsentimental",
                                       'Sentiment analysis dataset': "\\datasets\\Twitter US Airline Sentiment "
                                                                     "dataset.csv",
                                       'language': 'english'
                                       },
            'Russian with LSTM sholo don': {'tasks': ['Authorship analysis'],
                                            'base language model': "\\savedModels\\russianLSTMNextWord",
                                            'Authorship analysis model': "\\savedModels\\russianLSTMNextWord",
                                            'Authorship analysis dataset': "\\datasets\\And_Quiet_Flows_the_Don_1.txt",
                                            'language': 'russian'
                                            },
            'Russian with GRU sholo don': {'tasks': ['Authorship analysis'],
                                           'base language model': "\\savedModels\\russianGRUNextWord",
                                           'Authorship analysis model': "\\savedModels\\russianGRUNextWord",
                                           'Authorship analysis dataset': "\\datasets\\And_Quiet_Flows_the_Don_1.txt",
                                           'language': 'russian'
                                           },
        }
        sh.close()

    def update_model_task_dict(self, model_task_dict):
        sh = shelve.open(ROOT_DIR + "\\database\\model_task_dict_AppShelveDb")
        sh['model_task_dict'] = model_task_dict
        sh.close()

    def get_embedding_options_dict(self):
        sh = shelve.open(ROOT_DIR + "\\database\\embedding_options_dict_AppShelveDb")
        sentiment_dict = sh['embedding_options_dict']
        sh.close()
        return sentiment_dict

    def set_embedding_options_dict(self):
        sh = shelve.open(ROOT_DIR + "\\database\\embedding_options_dict_AppShelveDb")
        sh['embedding_options_dict'] = {
            "LaBSE_15": ["english", "russian"],
            "LaBSE_109": ["english", "russian"],
            "Bert": ["english"],
        }
        sh.close()

    def update_embedding_options_dict(self, embedding_options_dict):
        sh = shelve.open(ROOT_DIR + "\\database\\embedding_options_dict_AppShelveDb")
        sh['embedding_options_dict'] = embedding_options_dict
        sh.close()


"""
MiscMethodsShelveDb get,set,and update shelves file that contains dictionarys which are used in MiscMethods class
"""


class MiscMethodsShelveDb:
    def __init__(self):
        super(MiscMethodsShelveDb, self).__init__()

    def get_sentiment_dict(self):
        sh = shelve.open(ROOT_DIR + "\\database\\sentiment_dict_MiscMethodsShelveDb")
        sentiment_dict = sh['sentiment_dict']
        sh.close()
        return sentiment_dict

    def set_sentiment_dict(self):
        sh = shelve.open(ROOT_DIR + "\\database\\sentiment_dict_MiscMethodsShelveDb")
        sh['sentiment_dict'] = {
            1: "neutral",
            0: "negative",
            2: "positive"
        }
        sh.close()

    def update_sentiment_dict(self, sentiment_dict):
        sh = shelve.open(ROOT_DIR + "\\database\\sentiment_dict_MiscMethodsShelveDb")
        sh['sentiment_dict'] = sentiment_dict
        sh.close()


"""
This main is initiate the data base to its initial values.
In case of need for change the initial database, the "set" methods should be updated and than 
this main should be executed.
"""

if __name__ == '__main__':
    model_shelve_db = ModelShelveDb()
    dataset_preprocess_shelve_db = DatasetPreprocessShelveDb()
    app_shelve_db = AppShelveDb()
    misc_methods_shelve_sb = MiscMethodsShelveDb()

    model_shelve_db.set_url_for_embedding()
    dataset_preprocess_shelve_db.set_sentiment_dict()
    dataset_preprocess_shelve_db.set_regex_for_language()
    app_shelve_db.set_model_task_dict()
    app_shelve_db.set_embedding_options_dict()
    misc_methods_shelve_sb.set_sentiment_dict()
