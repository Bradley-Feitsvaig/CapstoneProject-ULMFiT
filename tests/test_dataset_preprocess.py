import os
from unittest import TestCase

from dataset_preprocess import DatasetPreprocess
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDatasetPreprocess(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_preprocess = DatasetPreprocess()
    """
    checks if the number of sequences is equal to the number of labels after dividing the raw text.
    """
    def test_divide_text_file_into_sequences_and_next_words(self):
        sequences, temp_labels, sentences = self.dataset_preprocess.divide_text_file_into_sequences_and_next_words(file_path=ROOT_DIR + '\\..\\datasets\\put.txt', language='russian', number_of_samples=8000)
        self.assertEqual(len(sequences), len(temp_labels))
    """
    checks if the number of sentiment sentences is equal to the number of labels and there are exactly 3 classes of predictions.
    """
    def test_build_data_set_for_sentimental_analysis(self):
        clear_sentiment_sentences, sentiment_labels, number_of_target_class_clasees = self.dataset_preprocess.build_data_set_for_sentimental_analysis(ROOT_DIR + '\\..\\datasets\\Twitter US Airline Sentiment dataset.csv', 'english', 'text', 'sentiment')
        self.assertEqual(len(clear_sentiment_sentences), len(sentiment_labels))
        self.assertEqual(number_of_target_class_clasees,3)
    """
    checks if the returned value is of type "list".
    """
    def test_vectors_for_labels(self):
        sequences, temp_labels, sentences = self.dataset_preprocess.divide_text_file_into_sequences_and_next_words(
            file_path=ROOT_DIR + '\\..\\datasets\\put.txt', language='russian', number_of_samples=8000)
        word_index, total_unique_words = self.dataset_preprocess.tokenization(sentences)
        labels = self.dataset_preprocess.vectors_for_labels(word_index, total_unique_words,temp_labels)
        self.assertTrue(isinstance(labels,list))

    """
    checks if the number of sequences is equal to the number of labels after building the dataset for training the model.
    """
    def test_build_dataset_for_predicting_the_next_word(self):
        sequences, labels, total_unique_words = self.dataset_preprocess.build_dataset_for_predicting_the_next_word(file_path=ROOT_DIR + '\\..\\datasets\\put.txt', language='russian', number_of_samples=8000)
        self.assertEqual(len(sequences), len(labels))
