import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from shelve_db import MiscMethodsShelveDb
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class MiscMethods:
    def __init__(self):
        super(MiscMethods, self).__init__()
        misc_methods_shelve_sb = MiscMethodsShelveDb()
        self.sentiment_dict = misc_methods_shelve_sb.get_sentiment_dict()

    def get_loss_plot(self, history, plot_dir_name):  # Generate loss plot and saves it in output directory
        plt.clf()
        loss_train = history.history['loss']
        plt.plot(loss_train, 'r', label='Training loss')
        plt.title('loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(plot_dir_name)

    def get_accuracy_plot(self, history, plot_dir_name):  # Generate accuracy plot and saves it in output directory
        plt.clf()
        loss_train = history.history['accuracy']
        plt.plot(loss_train, 'g', label='Training accuracy')
        plt.title('accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(plot_dir_name)

    """
    Builds scatter plot with 2 lists of vectors, one for training predictions and ont for testing predictions,
    this plot shows 2 clusters of 2 different books for authorship analysis.
    """
    def get_scatter_plot(self, train_predictions, test_predictions):
        plt.clf()
        t_sne = TSNE(n_components=2)  # Linear dimensionality reduction with tsne
        x_embedded = t_sne.fit_transform(train_predictions)
        t = x_embedded.transpose()
        plt.scatter(t[0], t[1], color='red', alpha=0.5, label='First book (train predictions)')
        t_sne = TSNE(n_components=2)
        x_embedded = t_sne.fit_transform(test_predictions)
        t = x_embedded.transpose()
        plt.scatter(t[0], t[1], color='blue', alpha=0.5, label='Second book (test predictions)')
        plt.legend()
        plt.savefig(ROOT_DIR + '\\outputs\\scatter_plot.jpg')

    def get_sentiment_analysis_prediction(self, prediction_vector):  # returns text value of sentiment
        max_value = max(prediction_vector)
        max_index = prediction_vector.index(max_value)
        return self.sentiment_dict[max_index]
