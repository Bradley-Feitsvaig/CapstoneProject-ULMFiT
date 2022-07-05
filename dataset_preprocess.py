import re
import pandas as pd
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from shelve_db import DatasetPreprocessShelveDb


class DatasetPreprocess:
    def __init__(self):
        super(DatasetPreprocess, self).__init__()
        dataset_preprocess_shelve_db = DatasetPreprocessShelveDb()
        self.regex_for_language = dataset_preprocess_shelve_db.get_regex_for_language()  # Dictionary of regular expression for wanted language.
        self.sentiment_dict = dataset_preprocess_shelve_db.get_sentiment_dict()

    """
    Divide the text into sequences and the next words.
    Example:
    
    sentence = 'the boy who lived'.
    
    sequence=['the boy who'], next word='lived'.
    """
    def divide_text_file_into_sequences_and_next_words(self, file_path, language, number_of_samples):
        data = pd.read_fwf(file_path, sep=" ", header=None, dtype="string", widths=[100])
        data = data[0]
        data = data.sample(n=min([len(data), number_of_samples]))
        data = data.str.lower()
        stop_words = set(stopwords.words(language))
        data = data.apply(
            lambda x: ' '.join([word for word in x.split() if word not in stop_words]))  # remove stop words
        text = ""
        for line in data:
            text += line
        text = re.sub(self.regex_for_language[language], '', text)
        text = re.sub(' +', ' ', text)
        sentences = text.split('.')
        sentences = [x for x in sentences if x]
        sequences = []
        temp_labels = []
        for sentence in sentences:
            sentence = sentence.strip()
            temp_sentence = sentence.split(" ")
            for i in range(len(temp_sentence)):
                s = [' '.join(temp_sentence[0:i + 1])]
                sequences.append(s)
                temp_labels.append(temp_sentence[i])
        temp_labels.pop(0)  # Shift left the temp labels list (temp_labels is a list with the next word for each
        # sequence in the text)
        sequences.pop(-1)  # delete the last sequences (there is no next word for it)
        sequences = [x for x in sequences if x != ['']]
        temp_labels = [x for x in temp_labels if x != '']
        return sequences, temp_labels, sentences

    """
    Build data set for sentimental analysis
    Build vector for each label in the sentimental analysis dataset
    vectors: [1,0,0] = negative ,[0,1,0] = neutral ,[0,0,1] = positive
    """
    def build_data_set_for_sentimental_analysis(self, file_path, language, text_column_name, sentiment_column_name):
        df = pd.read_csv(file_path, dtype="string")
        df = df.sample(n=min([len(df), 8000]))
        stop_words = set(stopwords.words(language))
        df[text_column_name] = df[text_column_name].str.lower()
        stop_words = set(stopwords.words(language))
        df[text_column_name] = df[text_column_name].apply(
            lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        sentiment_labels = []
        number_of_target_class_clasees = len(set(df[sentiment_column_name]))
        for sentiment in df[sentiment_column_name]:
            if sentiment not in self.sentiment_dict.keys():
                continue
            sentiment_label = [0, 0, 0]
            sentiment_label[self.sentiment_dict[sentiment]] = 1
            sentiment_labels.append(sentiment_label)
        sentiment_sentences = df[text_column_name].values.tolist()
        clear_sentiment_sentences = []
        for sentence in sentiment_sentences:
            sentence = re.sub(self.regex_for_language[language], '', sentence)
            clear_sentiment_sentences.append(sentence)
        return clear_sentiment_sentences, sentiment_labels, number_of_target_class_clasees

    """
    Text tokenization using keras Tokenizer.
    """
    def tokenization(self, sentences):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)  # After the Tokenizer has been created, we then fit it on the data.
        word_index = tokenizer.word_index  # word index maps words in our vocabulary to their numeric representation.
        total_unique_words = len(word_index) + 1  # number of words in vocabulary
        return word_index, total_unique_words

    """
    Create vector for each label.
    (label is the next word for each sequence)
    """
    def vectors_for_labels(self, word_index, total_unique_words, temp_labels):
        labels = []
        for i in range(len(temp_labels)):
            lst = [0] * total_unique_words
            lst[word_index[temp_labels[i]]] = 1
            labels.append(lst)
        return labels

    """
    Method builds the data set for prediction of the next word for a sequence, it uses the previews methods
    """
    def build_dataset_for_predicting_the_next_word(self, file_path, language, optional_file_path=None,
                                                   number_of_samples=8000):
        if optional_file_path:
            samples = number_of_samples // 2
        else:
            samples = number_of_samples
        sequences, temp_labels, sentences = self.divide_text_file_into_sequences_and_next_words(file_path, language,
                                                                                                samples)
        if optional_file_path:
            optional_sequences, optional_temp_labels, optional_sentences = self.divide_text_file_into_sequences_and_next_words(
                optional_file_path, language, samples)
            word_index, total_unique_words = self.tokenization(sentences + optional_sentences)
            optional_labels = self.vectors_for_labels(word_index, total_unique_words, optional_temp_labels)
            labels = self.vectors_for_labels(word_index, total_unique_words, temp_labels)
            return sequences, labels, total_unique_words, optional_sequences, optional_labels
        else:
            word_index, total_unique_words = self.tokenization(sentences)
            labels = self.vectors_for_labels(word_index, total_unique_words, temp_labels)
            return sequences, labels, total_unique_words
